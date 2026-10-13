"""Which GPU classes can run a model: the New session form greys out the ones that cannot, and the Models page
shows the same verdicts. Blocked means the runtime cannot run the configuration at all, not merely slowly."""

from unittest.mock import AsyncMock, Mock

import pytest

from colabinet.alerts import Alerts, NullNotifier
from colabinet.config import GPU_CLASSES, REPO_ROOT, SYSTEM_RAM_GB, Settings
from colabinet.presets import load_presets
from colabinet.supervisor import CreateRefused, CreateRequest, Supervisor
from colabinet.web.views import default_gpu, fit_matrix, gpu_fit, session_fit

PRESETS = load_presets(REPO_ROOT / "presets.toml").presets


def fits(preset_id, gpus=GPU_CLASSES, ram=SYSTEM_RAM_GB, **overrides):
    preset = PRESETS[preset_id]
    return session_fit(preset.engine, preset, list(gpus), ram, **overrides)


def statuses(preset_id, **overrides):
    return {gpu: row["status"] for gpu, row in fits(preset_id, **overrides).items()}


@pytest.mark.parametrize("original, abliterated", [
    ("qwen38-27b-q4km", "qwen38-27b-abl-q4k"),
    ("qwen38-27b-exl3-4bpw", "qwen38-27b-abl-exl3-4bpw"),
])
def test_original_qwen_presets_pair_with_the_abliterated_ones(original, abliterated):
    plain, abl = PRESETS[original], PRESETS[abliterated]
    assert "abliterated" not in f"{plain.repo} {plain.label} {plain.alias}".lower()
    assert plain.repo.split("/")[-1].startswith("Qwen3.8-27B")
    assert (plain.engine, plain.template, plain.gpu, plain.kv_bytes_per_token_f16, plain.parallel) == (
        abl.engine, abl.template, abl.gpu, abl.kv_bytes_per_token_f16, abl.parallel)
    assert plain.licence.startswith("Apache-2.0")
    assert statuses(original) == statuses(abliterated)


def test_original_exl3_uses_the_same_bitrate_branch_and_offloaded_vision():
    plain = PRESETS["qwen38-27b-exl3-4bpw"]
    assert plain.repo == "turboderp/Qwen3.8-27B-exl3" and plain.revision == "4.00bpw"
    assert plain.vision and plain.vision_offload
    assert PRESETS["qwen38-27b-q4km"].mmproj


def test_exllamav3_needs_a_gpu_with_room_for_weights_and_cache():
    assert statuses("qwen38-27b-exl3-4bpw") == {"CPU": "blocked", "T4": "blocked", "T4-HIGHMEM": "blocked",
                                                 "L4": "ok", "A100": "ok", "A100-80": "ok", "G4": "ok"}
    t4 = fits("qwen38-27b-exl3-4bpw")["T4"]
    assert t4["label"] == "Exceeds VRAM"
    # Measured on a T4: even a Q8 cache at 32K failed "Insufficient VRAM in split for model and cache".
    assert statuses("qwen38-27b-exl3-4bpw", kv="Q8")["T4"] == "blocked"
    # Overrides are still re-evaluated: a much shorter context is not ruled out (unmeasured, so only a warning).
    assert statuses("qwen38-27b-exl3-4bpw", kv="Q8", ctx=4096)["T4"] != "blocked"


def test_llamacpp_offload_stays_available_until_vram_and_host_ram_run_out():
    q4 = fits("qwen38-27b-q4km")
    assert q4["CPU"]["status"] == "blocked" and q4["CPU"]["label"] == "Too large"
    assert q4["T4"]["status"] == "warn" and q4["T4"]["label"] == "CPU offload required"
    assert q4["L4"]["status"] == "warn" and q4["A100"]["status"] == "ok"
    flash = statuses("qwen38-flash-next-abl-q4k-xl")
    assert [gpu for gpu, status in flash.items() if status != "blocked"] == ["A100", "A100-80", "G4"]


def test_recorded_host_ram_decides_offload_room():
    preset = PRESETS["qwen38-27b-q4km"]
    roomy = session_fit("llamacpp", preset, ["CPU"], {"CPU": 32.0})["CPU"]
    assert roomy["status"] == "warn" and roomy["label"] == "CPU only"
    tight = gpu_fit("llamacpp", "L4", PRESETS["qwen38-flash-next-abl-q4k-xl"], 111.3, None, 8192, "f16", 52.0)
    assert tight["status"] == "blocked" and "VRAM and host RAM" in tight["message"]


@pytest.mark.parametrize("preset_id", ["qwen38-flash-next-strata-iq2-xs", "qwen38-flash-next-strata-iq3-xxs",
                                       "qwen38-flash-next-strata-iq3-s"])
def test_strata_follows_upstreams_host_ram_policy(preset_id):
    # Measured 2026-10-13: setup refused every quant on a T4 (13.6 GB RAM) and ran them on T4-HIGHMEM.
    strata = fits(preset_id)
    assert [gpu for gpu, row in strata.items() if row["status"] == "blocked"] == ["CPU", "T4"]
    assert strata["T4"]["label"] == "Not enough RAM" and "low-RAM policy" in strata["T4"]["message"]


def test_strata_low_ram_args_decide_whether_iq3_fits_an_l4():
    # Measured on an L4 (56.9 GB): auto mode refused IQ3_XXS "requires about 60 GiB host RAM". The L4 preset
    # defaults force low-RAM mode, and overriding engine args brings the refusal back with a hint.
    preset = PRESETS["qwen38-flash-next-strata-iq3-xxs"]
    assert preset.defaults_for("L4").extra_args == ("--low-ram", "on")
    assert statuses("qwen38-flash-next-strata-iq3-xxs")["L4"] == "ok"
    auto = fits("qwen38-flash-next-strata-iq3-xxs", gpus=["L4"], extra_args=("--low-ram", "auto"))["L4"]
    assert auto["status"] == "blocked" and "requires about 60 GiB" in auto["message"]
    assert "--low-ram on" in auto["message"]


def test_custom_models_are_only_blocked_where_their_engine_or_known_size_rules_them_out():
    unknown = session_fit("exllamav3", None, list(GPU_CLASSES), SYSTEM_RAM_GB)
    assert unknown["CPU"]["status"] == "blocked" and unknown["T4"]["label"] == "Size unknown"
    assert all(row["status"] != "blocked" for row in session_fit("llamacpp", None, list(GPU_CLASSES),
                                                                   SYSTEM_RAM_GB).values())
    sized = session_fit("llamacpp", None, ["CPU", "L4"], SYSTEM_RAM_GB, weights_gb=40.0)
    assert sized["CPU"]["status"] == "blocked" and sized["L4"]["status"] == "warn"


def test_new_session_starts_on_l4_unless_it_cannot_run_the_model():
    assert default_gpu(fits("qwen38-27b-q4km")) == "L4"
    assert default_gpu(fits("qwen38-27b-exl3-4bpw")) == "L4"
    assert default_gpu(fits("qwen38-flash-next-abl-q4k-xl")) == "A100"
    # Nothing offered can run it: keep the usual default and let the form flag it.
    assert default_gpu(fits("qwen38-flash-next-abl-q4k-xl", gpus=["CPU", "L4"])) == "L4"
    # Without L4 the cheapest offered GPU is the usual default.
    assert default_gpu(fits("qwen38-27b-exl3-4bpw", gpus=["CPU", "T4", "A100", "G4"])) == "A100"
    assert default_gpu(fits("qwen38-27b-q4km", gpus=["CPU", "T4", "A100"])) == "T4"


def test_models_page_matrix_uses_the_same_verdicts():
    matrix = {row["gpu"]: row for row in fit_matrix(PRESETS["qwen38-flash-next-abl-q4k-xl"], SYSTEM_RAM_GB)}
    assert matrix["L4"]["label"] == "Too large" and matrix["L4"]["variant"] == "destructive"
    assert matrix["A100"]["status"] == "warn" and matrix["A100"]["kv_unknown"]


@pytest.fixture
def supervisor(store, clock, tmp_path, monkeypatch):
    sup = Supervisor(settings=Settings(data_dir=tmp_path), store=store, cli=None, tunnels=None, proxies=None,
                     alerts=Alerts(NullNotifier(), clock.now), clock=clock,
                     presets=load_presets(REPO_ROOT / "presets.toml"), http=object())
    monkeypatch.setattr(sup, "_start_driver", Mock())
    monkeypatch.setattr(sup, "_cancel_driver", AsyncMock())
    return sup


@pytest.mark.parametrize("request_fields", [
    {"preset_id": "qwen38-27b-exl3-4bpw"},
    {"engine": "exllamav3", "repo": "turboderp/Qwen3.8-27B-exl3", "revision": "4.00bpw"},
])
async def test_exllamav3_on_a_cpu_runtime_is_refused_before_listing_or_billing(supervisor, monkeypatch,
                                                                              request_fields):
    monkeypatch.setattr(supervisor, "cap_today", AsyncMock(side_effect=AssertionError("billing reached")))
    monkeypatch.setattr(supervisor, "fresh_listing", AsyncMock(side_effect=AssertionError("listing reached")))
    with pytest.raises(CreateRefused, match="ExLlamaV3 requires a GPU runtime"):
        await supervisor.create(CreateRequest(name="exl", gpu_class="CPU", confirm=True, **request_fields))
    assert await supervisor.store.sessions() == []
