import json

import pytest

from colabinet.config import REPO_ROOT, SEED_RATES, SYSTEM_RAM_GB
from colabinet.presets import load_presets
from colabinet.web.app import make_templates
from colabinet.web.views import session_fit
from tests.fakes.browser import run_browser

BASE = {"balance": None, "burn_rate": 0, "live_count": 0, "alert_count": 0,
        "command_sessions": [], "daily": {"spend": 0, "cap": 10}}


@pytest.mark.parametrize("preset_id", ["qwen38-27b-abl-q4k", "qwen38-27b-abl-exl3-4bpw",
                                      "qwen38-flash-next-strata-iq2-xs", "custom"])
def test_new_session_fields_follow_engine_and_exclude_hidden_values(tmp_path, preset_id):
    presets = list(load_presets(REPO_ROOT / "presets.toml").presets.values())
    selected = next((preset for preset in presets if preset.id == preset_id), None)
    page = make_templates().env.get_template("pages/new.html").render(
        csrf="test-csrf", active="new", base=BASE, presets=presets, preset=preset_id,
        engine=selected.engine if selected else "llamacpp", gpu="L4", gpus=["CPU", "L4"],
        rates={gpu: rate for gpu, (rate, _) in SEED_RATES.items()}, idle_default=30,
    )
    result = run_browser(tmp_path, page, r"""
return await evaluate(`(() => {
  const form = document.getElementById('new-form');
  const preset = document.getElementById('preset');
  const engine = document.getElementById('engine');
  const kv = document.getElementById('kv');
  const extra = document.getElementById('extra_args');
  const change = (element, value) => {
    element.value = value;
    element.dispatchEvent(new Event('change', { bubbles: true }));
  };
  const snapshot = () => ({
    visible: ['engine', 'repo', 'file', 'revision', 'alias'].filter((id) =>
      document.getElementById(id).getBoundingClientRect().height > 0),
    submitted: Array.from(new FormData(form).keys()),
    kv: kv.value, extra: extra.value,
    projection: document.getElementById('refusal_projection').value,
    projectionEnabled: !document.getElementById('refusal_projection').disabled,
  });
  const initial = snapshot();
  change(preset, 'custom');
  change(engine, 'llamacpp');
  document.getElementById('file').value = 'weights.gguf';
  document.getElementById('revision').value = '4.00bpw';
  kv.value = 'q4_0'; extra.value = '--reasoning-budget 4096';
  const llama = snapshot();
  change(engine, 'exllamav3');
  const exl = snapshot();
  kv.value = 'Q8'; extra.value = 'model.chunk_size=4096';
  change(engine, 'strata');
  change(document.getElementById('refusal_projection'), 'true');
  const strata = snapshot();
  change(engine, 'exllamav3');
  kv.value = 'Q8'; extra.value = 'model.chunk_size=4096';
  change(preset, 'qwen38-27b-abl-exl3-4bpw');
  const exlPreset = snapshot();
  change(preset, 'qwen38-27b-abl-q4k');
  const llamaPreset = snapshot();
  change(preset, 'custom');
  const customAgain = snapshot();
  document.dispatchEvent(new Event('cb:pageEnter'));
  return { initial, llama, exl, strata, exlPreset, llamaPreset, customAgain, reentered: snapshot() };
})()`);
""")
    initial = result["initial"]
    assert initial["visible"] == (["engine", "repo", "file", "alias"] if preset_id == "custom" else [])
    assert initial["projectionEnabled"] is bool(selected and selected.engine == "strata")
    llama, exl = result["llama"], result["exl"]
    assert llama["visible"] == ["engine", "repo", "file", "alias"]
    assert "file" in llama["submitted"] and "revision" not in llama["submitted"]
    assert exl["visible"] == ["engine", "repo", "revision", "alias"]
    assert "revision" in exl["submitted"] and "file" not in exl["submitted"]
    assert exl["kv"] == exl["extra"] == ""
    assert result["exlPreset"]["visible"] == []
    assert result["exlPreset"]["kv"] == "Q8" and result["exlPreset"]["extra"] == "model.chunk_size=4096"
    assert result["llamaPreset"]["kv"] == result["llamaPreset"]["extra"] == ""
    assert all(field not in result["llamaPreset"]["submitted"] for field in ("repo", "file", "revision", "alias"))
    assert result["customAgain"]["visible"] == ["engine", "repo", "file", "alias"]
    assert result["reentered"] == result["customAgain"]
    strata = result["strata"]
    assert strata["visible"] == ["engine", "repo", "file", "alias"]
    assert "file" in strata["submitted"] and "revision" not in strata["submitted"]
    assert strata["projectionEnabled"] and strata["projection"] == "true"
    assert "refusal_projection" in strata["submitted"]
    assert not result["exlPreset"]["projectionEnabled"]
    assert "refusal_projection" not in result["exlPreset"]["submitted"]


def test_gpu_choices_grey_out_unrunnable_classes_and_keep_an_explicit_choice(tmp_path):
    """An explicit CPU pick with an ExLlamaV3 model stays selected and submittable, is flagged with an alternative,
    and the review's per-class fit greys out the rest; accepting the alternative moves the selection."""
    templates = make_templates().env
    presets = list(load_presets(REPO_ROOT / "presets.toml").presets.values())
    preset = next(item for item in presets if item.id == "qwen38-27b-exl3-4bpw")
    gpus = ["CPU", "T4", "L4", "A100"]
    fits = session_fit(preset.engine, preset, gpus, SYSTEM_RAM_GB)
    rates = {gpu: rate for gpu, (rate, _) in SEED_RATES.items()}
    page = templates.get_template("pages/new.html").render(
        csrf="test-csrf", active="new", base=BASE, presets=presets, preset=preset.id, engine=preset.engine,
        gpu="CPU", gpus=gpus, rates=rates, idle_default=30, gpu_fit=fits)

    def review(gpu, suggest):
        return templates.get_template("partials/new_estimate.html").render(
            fit=fits[gpu], fit_display=fits[gpu], gpu_fit=fits, suggest=suggest, gpu=gpu, engine=preset.engine,
            ctx=fits[gpu]["ctx"], kv=fits[gpu]["kv"], preset=preset, refusal_projection=False,
            daily={"spend": 0, "cap": 10},
            cost={"rate": rates[gpu], "idle": 30, "cap": None, "idle_waste": 0.5, "cap_minutes": None,
                  "endpoint": "next free", "no_limit": False})

    result = run_browser(tmp_path, page, r"""
await evaluate(`document.addEventListener('htmx:beforeRequest', (event) => event.preventDefault())`);
const swap = (markup) => evaluate(`(() => {
  const target = document.getElementById('estimate');
  target.innerHTML = ${JSON.stringify(markup)};
  target.dispatchEvent(new CustomEvent('htmx:afterSwap', { bubbles: true }));
  return true;
})()`);
const state = () => evaluate(`(() => ({
  checked: [...document.querySelectorAll('#new-form input[name="gpu"]:checked')].map((input) => input.value),
  disabled: [...document.querySelectorAll('#new-form input[name="gpu"]:disabled')].map((input) => input.value),
  submitted: new FormData(document.getElementById('new-form')).getAll('gpu'),
  labels: Object.fromEntries([...document.querySelectorAll('.cb-gpu')].map((card) =>
    [card.querySelector('input').value, card.querySelector('.cb-gpu-fit').textContent])),
  note: document.getElementById('gpu-note').hidden ? null : document.getElementById('gpu-note').textContent,
  bar: document.getElementById('create-bar-summary').textContent,
  // The cards draw their own radio; the native one must stay invisible, greyed out or not.
  nativeRadiosShown: [...document.querySelectorAll('#new-form input[name="gpu"]')]
    .filter((input) => getComputedStyle(input).opacity !== '0').map((input) => input.value),
}))()`);
const initial = await state();
await swap(BLOCKED_REVIEW);
const flagged = await state();
await evaluate(`document.querySelector('#gpu-note button').click(); true`);
const moved = await state();
await swap(L4_REVIEW);
const settled = await state();
return { initial, flagged, moved, settled };
""".replace("BLOCKED_REVIEW", json.dumps(review("CPU", "L4"))).replace("L4_REVIEW", json.dumps(review("L4", ""))))
    initial, flagged = result["initial"], result["flagged"]
    assert initial["checked"] == initial["submitted"] == ["CPU"]
    assert initial["disabled"] == ["T4"] and initial["labels"]["CPU"] == "GPU required"
    assert flagged["checked"] == flagged["submitted"] == ["CPU"] and flagged["disabled"] == ["T4"]
    assert flagged["note"].startswith("CPU cannot run this model") and flagged["note"].endswith("Use L4 instead")
    assert flagged["bar"] == "CPU cannot run this model. Choose another GPU."
    assert result["moved"]["checked"] == ["L4"]
    settled = result["settled"]
    assert settled["checked"] == ["L4"] and settled["disabled"] == ["CPU", "T4"] and settled["note"] is None
    assert settled["bar"].startswith("L4 at") and settled["labels"]["L4"] == "Fits in VRAM"
    assert all(state["nativeRadiosShown"] == [] for state in (initial, flagged, settled))


@pytest.mark.parametrize("preset_id, gpu, variant", [
    ("qwen38-flash-next-abl-q4k-xl", "L4", "destructive"),  # cannot run: greyed out
    ("qwen38-flash-next-abl-q4k-xl", "A100", "warning"),  # runs with CPU offload: amber, as on its GPU card
    ("qwen38-27b-exl3-4bpw", "T4", "destructive"),
    ("qwen38-27b-exl3-4bpw", "L4", "success"),
    ("qwen38-flash-next-strata-iq3-s", "G4", "success"),  # GPU tested
])
def test_review_alert_uses_the_gpu_choices_status(preset_id, gpu, variant):
    preset = load_presets(REPO_ROOT / "presets.toml").presets[preset_id]
    fit = session_fit(preset.engine, preset, [gpu], SYSTEM_RAM_GB)[gpu]
    html = make_templates().env.get_template("partials/new_estimate.html").render(
        fit=fit, fit_display=fit, gpu_fit={gpu: fit}, suggest="", gpu=gpu, engine=preset.engine, ctx=fit["ctx"],
        kv=fit["kv"], preset=preset, refusal_projection=False, daily={"spend": 0, "cap": 10},
        cost={"rate": 1.0, "idle": 30, "cap": None, "idle_waste": 0.5, "cap_minutes": None, "endpoint": "next free",
              "no_limit": False})
    assert f'<div class="alert" data-variant="{variant}">' in html
