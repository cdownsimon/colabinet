import pytest

from colabinet.activity import BYTE_WINDOW_S, INFLIGHT_MAX_S, Activity, is_inference
from colabinet.alerts import Alerts, NullNotifier
from colabinet.config import REPO_ROOT
from colabinet.guards import evaluate, evaluate_daily
from colabinet.presets import PresetError, load_presets


def test_repo_presets_load():
    p = load_presets(REPO_ROOT / "presets.toml")
    q = p.presets["qwen38-27b-abl-q4k"]
    assert p.engines["llamacpp"]["release"] == "b11273"
    assert q.defaults_for("L4").kv == "q4_0"
    assert q.defaults_for("A100-80").kv == "f16"
    assert q.defaults_for("T4-HIGHMEM").ctx == 32768  # no T4 entry -> generic defaults
    # Measured on real runtimes of each class; a class that cannot run the preset has no figure.
    assert q.measured["L4"].gen_tps == 13.1 and q.measured["L4"].prefill_tps == 660
    assert "CPU" not in q.measured
    assert all(preset.measured for preset in p.presets.values())


@pytest.mark.parametrize(
    "text,msg",
    [
        ("[[preset]]\nid='a'\nengine='llamacpp'\nlabel='x'\n", "missing 'repo'"),
        ("[[preset]]\nid='a'\nengine='vllm'\nlabel='x'\nrepo='r'\n", "unknown engine"),
        ("[[preset]]\nid='a'\nengine='llamacpp'\nlabel='x'\nrepo='r'\n", "need 'file'"),
        ("[[preset]\n", "presets.toml"),
        ("[[preset]]\nid='a'\nengine='llamacpp'\nlabel='x'\nrepo='r'\nfile='f.gguf'\n[preset.gpu.L4]\nextra_args='-fa'\n",
         "extra_args must be a list of strings"),
        ("[[preset]]\nid='a'\nengine='llamacpp'\nlabel='x'\nrepo='r'\nfile='f.gguf'\n[preset.measured]\nL4={gen_tps=0}\n",
         "gen_tps and prefill_tps must be positive"),
        ("[[preset]]\nid='a'\nengine='llamacpp'\nlabel='x'\nrepo='r'\nfile='f.gguf'\n[preset.measured]\nL4={tps=3}\n",
         "unexpected keyword"),
    ],
)
def test_preset_errors(tmp_path, text, msg):
    f = tmp_path / "presets.toml"
    f.write_text(text)
    with pytest.raises(PresetError, match=msg):
        load_presets(f)


def sess(**over):
    s = dict(state="ready", state_since=0.0, created_at=0.0, stopped_at=None, rate_cu_h=6.0, cu_cap=None,
             deadline=None, idle_min=30, last_activity=0.0)
    s.update(over)
    return s


def test_idle_warn_then_stop():
    assert evaluate(sess(), now=24 * 60, activity_last=0, grace_s=600).action == "none"
    d = evaluate(sess(), now=26 * 60, activity_last=0, grace_s=600)
    assert (d.action, d.reason) == ("warn", "idle") and "4 min" in d.message
    assert evaluate(sess(), now=30 * 60, activity_last=0, grace_s=600).reason == "idle"
    # activity resets the clock
    assert evaluate(sess(), now=30 * 60, activity_last=20 * 60, grace_s=600).action == "none"


def test_idle_off_and_not_in_setup():
    assert evaluate(sess(idle_min=0), now=10**6, activity_last=0, grace_s=600).action == "none"
    assert evaluate(sess(state="setting_up"), now=10**6, activity_last=0, grace_s=600).action == "none"
    assert evaluate(sess(state="engine_down"), now=31 * 60, activity_last=0, grace_s=600).reason == "idle"


def test_cap_deadline_grace_precedence():
    assert evaluate(sess(cu_cap=1.0), now=600, activity_last=600, grace_s=600).reason == "cu_cap"
    assert evaluate(sess(deadline=100.0), now=100, activity_last=100, grace_s=600).reason == "deadline"
    assert evaluate(sess(state="setup_failed", state_since=0), now=600, activity_last=0, grace_s=600).reason == \
        "setup_failed"
    assert evaluate(sess(state="gpu_mismatch", state_since=0), now=599, activity_last=0, grace_s=600).action == "none"
    assert evaluate(sess(state="stopping", cu_cap=0.1), now=10**6, activity_last=0, grace_s=600).action == "none"


def test_daily_cap():
    assert evaluate_daily(7.9, 10).action == "none"
    assert evaluate_daily(8.0, 10).action == "warn"
    assert evaluate_daily(10.0, 10).reason == "daily_cap"
    assert evaluate_daily(99, 0).action == "none"


def test_activity_rules():
    assert is_inference("POST", "/v1/chat/completions/") and not is_inference("GET", "/v1/models")
    a = Activity(last=100.0)
    a.begin(1, 200.0)
    assert a.effective(200.0 + INFLIGHT_MAX_S) == 200.0 + INFLIGHT_MAX_S  # counts during the first hour
    a.touch(1, 200.0 + INFLIGHT_MAX_S)
    t = 200.0 + INFLIGHT_MAX_S + BYTE_WINDOW_S + 1
    assert a.effective(t) == 200.0 + INFLIGHT_MAX_S  # hung: no bytes for 5 min -> stops counting
    a.end(1, t, useful=False)
    assert a.effective(t) == 100.0
    a.end(2, 500.0, useful=True)
    assert a.effective(t) == 500.0


def test_alerts_dedupe_and_notify_critical_only():
    n = NullNotifier()
    alerts = Alerts(n, now=lambda: 5.0)
    alerts.raise_("stop_unconfirmed", "may still be billing", session_id=3)
    alerts.raise_("stop_unconfirmed", "still billing", session_id=3)
    alerts.raise_("idle_warn", "idle stop in 4 min", session_id=3)
    assert len(n.sent) == 1 and len(alerts.list()) == 2
    assert alerts.list()[0].message == "still billing"
    alerts.clear_session(3)
    assert alerts.list() == []
