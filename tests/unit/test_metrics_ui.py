from starlette.datastructures import FormData

from colabinet.metrics import GROUPS, MEASURES, QuerySpec
from colabinet.web.app import make_templates, spec_from_form
from tests.fakes.browser import run_browser


def test_visual_measure_order_preserves_existing_chart_and_export_order():
    form = FormData([("measures", "e2e_tps_p50"), ("measures", "ttft_p50"),
                     ("measures", "tokens_out"), ("measures", "requests"), ("measures", "cu")])
    assert spec_from_form(form)["measures"] == ["requests", "ttft_p50", "e2e_tps_p50", "tokens_out", "cu"]


def test_grouped_metrics_controls_preserve_saved_selections_and_keyboard_submission(tmp_path):
    spec = QuerySpec(group_by=["gpu", "model_kv", "client"],
                     measures=["requests", "gen_tps_p50", "ttft_p90", "tokens_out", "cu"])
    page = make_templates().env.get_template("pages/metrics.html").render(
        csrf="test-csrf", active="metrics", spec=spec, options={},
        groups=list(GROUPS), measures=list(MEASURES) + ["cu", "out_per_cu"],
        times=["24h", "7d", "30d", "all"], charts=["bar", "line", "table"], saved=[], saved_id=None,
        base={"balance": None, "burn_rate": 0, "live_count": 0, "alert_count": 0,
              "command_sessions": [], "daily": {"spend": 0, "cap": 10}},
    )
    result = run_browser(tmp_path, page, r"""
await evaluate(`document.addEventListener('htmx:beforeRequest', (event) => event.preventDefault())`);
const initial = await evaluate(`(() => {
  const form = document.getElementById('metrics-form');
  const data = new FormData(form);
  const options = Array.from(form.querySelectorAll('input[name="group_by"], input[name="measures"]'));
  return {
    groups: data.getAll('group_by'), measures: data.getAll('measures'),
    options: options.map((input) => ({ name: input.name, value: input.value,
      section: input.closest('fieldset')?.querySelector('legend')?.textContent.trim() })),
    prefillLabel: form.querySelector('[value="prefill_tps_p90"]').getAttribute('aria-label'),
  };
})()`);
await evaluate(`document.querySelector('[value="prefill_tps_p90"]').focus()`);
await send('Input.dispatchKeyEvent', { type: 'keyDown', key: ' ', code: 'Space', windowsVirtualKeyCode: 32 });
await send('Input.dispatchKeyEvent', { type: 'keyUp', key: ' ', code: 'Space', windowsVirtualKeyCode: 32 });
await evaluate(`document.querySelector('[name="group_by"][value="day"]').click()`);
const changed = await evaluate(`(() => {
  const data = new FormData(document.getElementById('metrics-form'));
  return { groups: data.getAll('group_by'), measures: data.getAll('measures') };
})()`);
return { initial, changed };
""")
    initial = result["initial"]
    assert set(initial["groups"]) == set(spec.group_by)
    assert set(initial["measures"]) == set(spec.measures)
    options = initial["options"]
    assert len(options) == len(GROUPS) + len(MEASURES) + 2
    assert {item["value"] for item in options if item["name"] == "group_by"} == set(GROUPS)
    assert {item["value"] for item in options if item["name"] == "measures"} == set(MEASURES) | {"cu", "out_per_cu"}
    sections = {item["value"]: item.get("section") for item in options}
    assert sections["gpu"] == sections["model_kv"] == "Model & hardware"
    assert sections["client"] == "Request details" and sections["day"] == "Time"
    assert sections["prefill_tps_p90"] == "Throughput" and sections["ttft_p90"] == "Latency"
    assert sections["tokens_out"] == "Tokens" and sections["cu"] == "Usage & cost"
    assert initial["prefillLabel"] == "Prefill P90, tokens per second"
    assert set(result["changed"]["groups"]) == set(spec.group_by) | {"day"}
    assert set(result["changed"]["measures"]) == set(spec.measures) | {"prefill_tps_p90"}


def test_filter_controls_show_live_selection_counts_and_preserve_saved_values(tmp_path):
    spec = QuerySpec(filters={"gpu": ["L4"], "model": ["archived/model"], "status_class": "5xx",
                              "exclude_playground": True}, chart="table")
    page = make_templates().env.get_template("pages/metrics.html").render(
        csrf="test-csrf", active="metrics", spec=spec,
        options={"gpu": ["L4", "G4"], "model": ["org/current-model"], "engine": ["llamacpp"],
                 "kv": ["q4_0"], "session": ["llm"], "client": ["curl/8.5"], "path": ["/v1/chat/completions"]},
        groups=list(GROUPS), measures=list(MEASURES) + ["cu", "out_per_cu"],
        times=["24h", "7d", "30d", "all"], charts=["bar", "line", "table"], saved=[], saved_id=None,
        base={"balance": None, "burn_rate": 0, "live_count": 0, "alert_count": 0,
              "command_sessions": [], "daily": {"spend": 0, "cap": 10}},
    )
    result = run_browser(tmp_path, page, r"""
await evaluate(`document.addEventListener('htmx:beforeRequest', (event) => event.preventDefault())`);
const initial = await evaluate(`(() => {
  const form = document.getElementById('metrics-form');
  const gpu = form.querySelector('[name="f_gpu"]').closest('details');
  return { entries: Array.from(new FormData(form).entries()), open: gpu.open,
    count: gpu.querySelector('[data-filter-count]')?.textContent.trim(),
    labels: ['status_class', 'chart'].map((name) =>
      form.querySelector('select[name="' + name + '"]').labels[0]?.textContent.trim()) };
})()`);
await evaluate(`document.querySelector('[name="f_gpu"]').closest('details').querySelector('summary').focus()`);
await send('Input.dispatchKeyEvent', {
  type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, text: '\r', unmodifiedText: '\r',
});
await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
const expanded = await evaluate(`document.querySelector('[name="f_gpu"]').closest('details').open`);
await evaluate(`new Promise((resolve, reject) => {
  const input = document.querySelector('[name="f_gpu"][value="G4"]');
  let attempts = 0;
  const focus = () => {
    input.focus();
    if (document.activeElement === input) resolve();
    else if (attempts++ < 60) requestAnimationFrame(focus);
    else reject(new Error('Expanded filter checkbox did not become focusable'));
  };
  focus();
})`);
await send('Input.dispatchKeyEvent', { type: 'keyDown', key: ' ', code: 'Space', windowsVirtualKeyCode: 32 });
await send('Input.dispatchKeyEvent', { type: 'keyUp', key: ' ', code: 'Space', windowsVirtualKeyCode: 32 });
const selected = await evaluate(`(() => {
  const filter = document.querySelector('[name="f_gpu"]').closest('details');
  return { count: filter.querySelector('[data-filter-count]')?.textContent.trim(),
    values: new FormData(document.getElementById('metrics-form')).getAll('f_gpu') };
})()`);
await evaluate(`(() => {
  document.querySelector('[name="f_gpu"][value="L4"]').click();
  document.querySelector('[name="f_gpu"][value="G4"]').click();
  document.querySelector('[name="exclude_playground"]').click();
  document.dispatchEvent(new Event('cb:pageEnter'));
})()`);
const cleared = await evaluate(`(() => {
  const filter = document.querySelector('[name="f_gpu"]').closest('details');
  return { count: filter.querySelector('[data-filter-count]')?.textContent.trim(),
    entries: Array.from(new FormData(document.getElementById('metrics-form')).entries()) };
})()`);
return { initial, expanded, selected, cleared };
""")
    initial = QuerySpec.parse(spec_from_form(FormData(result["initial"]["entries"])))
    assert initial.filters["gpu"] == ["L4"] and initial.filters["model"] == ["archived/model"]
    assert initial.filters["status_class"] == "5xx" and initial.filters["exclude_playground"]
    assert initial.chart == "table"
    assert result["initial"]["labels"] == ["Response status", "Chart type"]
    assert result["initial"]["count"] == "1 selected" and not result["initial"]["open"]
    assert result["expanded"]
    assert result["selected"] == {"count": "2 selected", "values": ["L4", "G4"]}
    cleared = QuerySpec.parse(spec_from_form(FormData(result["cleared"]["entries"])))
    assert "gpu" not in cleared.filters and not cleared.filters["exclude_playground"]
    assert cleared.filters["model"] == ["archived/model"]
    assert result["cleared"]["count"] == "All"
