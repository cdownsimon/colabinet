from colabinet.web.app import make_templates
from tests.fakes.browser import run_browser


def test_long_session_label_stays_inside_its_playground_grid_column(tmp_path):
    alias = "huihui.ai/qwen3.8-27b-abliterated-" + "long-model-name-" * 8
    html = make_templates().env.get_template("pages/playground.html").render(
        csrf="test-csrf", active="playground", selected=1,
        sessions=[{"id": 1, "colab_name": "coder", "proxy_port": 34001, "gpu_class": "L4", "run": {"alias": alias}}],
        base={"balance": None, "burn_rate": 0, "live_count": 1, "alert_count": 0,
              "command_sessions": [], "daily": {"spend": 0, "cap": 10}},
    )
    readings = run_browser(tmp_path, html, r"""
  const readings = [];
  for (const width of [1440, 1000, 800, 390, 360]) {
    await send('Emulation.setDeviceMetricsOverride', {
      width, height: 1000, deviceScaleFactor: 1, mobile: width < 560,
    });
    await evaluate(`new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))`);
    for (const language of ['python', 'curl']) {
      await evaluate(`(() => {
        const select = document.getElementById('pg-example-language');
        select.value = '${language}';
        select.dispatchEvent(new Event('change', { bubbles: true }));
      })()`);
      readings.push(await evaluate(`(() => {
      const rect = (node) => {
        const bounds = node.getBoundingClientRect();
        return { left: bounds.left, right: bounds.right, top: bounds.top, bottom: bounds.bottom };
      };
      return { width: innerWidth, documentWidth: document.documentElement.scrollWidth,
        examples: ['pg-${language}-panel', 'pg-example-language', 'pg-copy-example']
          .map((id) => ({ id, bounds: rect(document.getElementById(id)) })),
        controls: ['pg-session', 'pg-thinking', 'pg-max', 'pg-prompt'].map((id) => {
          const control = document.getElementById(id);
          return { id, bounds: rect(control), field: rect(control.closest('.field')),
            minWidth: getComputedStyle(control).minWidth,
            fieldMinWidth: getComputedStyle(control.closest('.field')).minWidth };
        }) };
    })()`));
    }
  }
  return readings;
""")
    for reading in readings:
        assert reading["documentWidth"] <= reading["width"], reading
        for example in reading["examples"]:
            assert example["bounds"]["left"] >= 0, (reading["width"], example)
            assert example["bounds"]["right"] <= reading["width"], (reading["width"], example)
        for control in reading["controls"]:
            assert control["bounds"]["left"] >= control["field"]["left"] - 0.5, (reading["width"], control)
            assert control["bounds"]["right"] <= control["field"]["right"] + 0.5, (reading["width"], control)
        session, thinking = [control["bounds"] for control in reading["controls"][:2]]
        assert session["right"] <= thinking["left"] or session["bottom"] <= thinking["top"], reading
