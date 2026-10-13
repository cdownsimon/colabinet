
(() => {
  const csrf = () => document.body.dataset.csrf;

  const nodeKey = (node) => node.nodeType === 1 ? node.id || node.dataset.cbKey || '' : '';
  const sameKind = (current, incoming) => current && current.nodeType === incoming.nodeType &&
    current.nodeName === incoming.nodeName && nodeKey(current) === nodeKey(incoming);
  const isHtmxAttribute = (name) => name.startsWith('hx-') || name.startsWith('data-hx-');
  const patchNode = (current, incoming, updates) => {
    if (current.isEqualNode(incoming)) return;
    if (current.nodeType !== 1) {
      if (current.nodeValue !== incoming.nodeValue) current.nodeValue = incoming.nodeValue;
      return;
    }
    let reinitialize = false;
    for (const attribute of Array.from(current.attributes)) {
      if (!incoming.hasAttribute(attribute.name) && attribute.name !== 'class') {
        current.removeAttribute(attribute.name);
        reinitialize ||= isHtmxAttribute(attribute.name);
      }
    }
    const transientClasses = ['htmx-request', 'htmx-swapping', 'htmx-settling'];
    const classes = new Set((incoming.getAttribute('class') || '').split(/\s+/).filter(Boolean));
    for (const name of transientClasses) if (current.classList.contains(name)) classes.add(name);
    const className = Array.from(classes).join(' ');
    if ((current.getAttribute('class') || '') !== className) {
      if (className) current.setAttribute('class', className); else current.removeAttribute('class');
    }
    for (const attribute of Array.from(incoming.attributes)) {
      if (attribute.name !== 'class' && current.getAttribute(attribute.name) !== attribute.value) {
        current.setAttribute(attribute.name, attribute.value);
        reinitialize ||= isHtmxAttribute(attribute.name);
      }
    }
    if (reinitialize) updates.changed.push(current);
    patchChildren(current, incoming, updates);
  };
  const patchChildren = (target, incoming, updates) => {
    const keyed = new Map(Array.from(target.childNodes).filter(nodeKey).map((node) => [nodeKey(node), node]));
    let cursor = target.firstChild;
    for (const child of Array.from(incoming.childNodes)) {
      const candidate = nodeKey(child) ? keyed.get(nodeKey(child)) : cursor;
      const retained = sameKind(candidate, child) ? candidate : child.cloneNode(true);
      if (retained !== cursor) {
        if (retained.parentNode === target && target.moveBefore) target.moveBefore(retained, cursor);
        else target.insertBefore(retained, cursor);
      }
      if (retained === candidate) patchNode(retained, child, updates);
      else if (retained.nodeType === 1) updates.added.push(retained);
      cursor = retained.nextSibling;
    }
    while (cursor) {
      const next = cursor.nextSibling;
      if (cursor.nodeType === 1) htmx.remove(cursor); else cursor.remove();
      cursor = next;
    }
  };
  const retainInteraction = (target, patch) => {
    const focused = target.contains(document.activeElement) ? document.activeElement : null;
    const selection = window.getSelection();
    const selected = selection?.rangeCount &&
      (target.contains(selection.anchorNode) || target.contains(selection.focusNode)) ? {
        anchor: selection.anchorNode, anchorOffset: selection.anchorOffset,
        focus: selection.focusNode, focusOffset: selection.focusOffset,
        start: selection.getRangeAt(0).startContainer, startOffset: selection.getRangeAt(0).startOffset,
        end: selection.getRangeAt(0).endContainer, endOffset: selection.getRangeAt(0).endOffset,
        text: selection.toString(),
      } : null;
    patch();
    if (focused?.isConnected && document.activeElement !== focused) focused.focus({ preventScroll: true });
    if (selected?.anchor.isConnected && selected.focus.isConnected) {
      const range = document.createRange();
      try {
        range.setStart(selected.start, selected.startOffset);
        range.setEnd(selected.end, selected.endOffset);
        if (range.toString() === selected.text && (selection.anchorNode !== selected.anchor ||
            selection.anchorOffset !== selected.anchorOffset || selection.focusNode !== selected.focus ||
            selection.focusOffset !== selected.focusOffset)) {
          selection.setBaseAndExtent(selected.anchor, selected.anchorOffset, selected.focus, selected.focusOffset);
        }
      } catch (_) {}
    }
  };
  if (window.htmx?.defineExtension) htmx.defineExtension('cb-patch', {
    isInlineSwap: (style) => style === 'cb-patch-outer',
    handleSwap: (style, target, fragment, settleInfo) => {
      const updates = { added: [], changed: [] };
      if (style === 'cb-patch') retainInteraction(target, () => patchChildren(target, fragment, updates));
      else if (style === 'cb-patch-outer' && sameKind(target, fragment.firstElementChild)) {
        retainInteraction(target, () => patchNode(target, fragment.firstElementChild, updates));
      } else return false;
      for (const node of updates.changed) settleInfo.tasks.push(() => htmx.process(node));
      return updates.added;
    },
  });

  window.cbToast = (category, title, description = '') => {
    const toaster = document.getElementById('toaster');
    if (toaster && typeof toaster.toast === 'function') toaster.toast({ category, title, description });
    else console.log(`[${category}] ${title} ${description}`);
  };
  document.addEventListener('cb-toast', (event) => {
    const timings = event.detail || {};
    cbToast(timings.category || 'info', timings.title || '', timings.description || '');
  });
  document.addEventListener('htmx:responseError', (event) => {
    let msg = event.detail.xhr.responseText;
    try { msg = JSON.parse(msg).error || msg; } catch (_) {}
    cbToast('error', 'Not done', msg);
  });
  document.addEventListener('htmx:sendError', () => cbToast('error', 'colabinet is not reachable', 'Is `colabinet serve` still running?'));

  let lastEvent = -1;
  const pollToasts = async () => {
    try {
      const response = await fetch(`/ui/toasts?after=${lastEvent}`);
      const data = await response.json();
      if (lastEvent >= 0) data.toasts.forEach((timings) => cbToast(timings.category, timings.title, timings.description));
      lastEvent = data.last;
    } catch (_) {}
  };

  window.cbCopy = async (text, title = 'Copied') => {
    try { await navigator.clipboard.writeText(text); cbToast('success', title, text.length > 80 ? '' : text); }
    catch (_) { cbToast('error', 'Could not copy', 'Select the text and copy it manually.'); }
  };
  window.cbConfirmStop = (id, name) => {
    const dialog = document.getElementById('stop-dialog');
    document.getElementById('stop-name').textContent = name;
    const btn = document.getElementById('stop-confirm');
    btn.setAttribute('hx-post', `/ui/actions/sessions/${id}/stop`);
    htmx.process(btn);
    dialog.showModal();
  };

  document.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      const dialog = document.getElementById('cmdk');
      if (dialog.open) dialog.close(); else { dialog.showModal(); document.getElementById('cmdk-input').focus(); }
    }
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && event.target?.id === 'pg-prompt') {
      event.preventDefault();
      if (!document.getElementById('pg-send')?.disabled) cbPlaygroundSend();
    }
  });

  const renderCharts = (root) => {
    if (!window.basecoat || !basecoat.chart || !window.Chart) return;
    root.querySelectorAll('canvas[data-cb-chart]:not([data-chart-initialized])').forEach((canvas) => {
      const data = JSON.parse(canvas.dataset.chartData || '[]');
      const keys = JSON.parse(canvas.dataset.chartSeries || '[]');
      // A series keeps its slot (e.g. by GPU class) so its colour does not change with what else is in range.
      const slots = JSON.parse(canvas.dataset.chartSlots || '[]');
      const series = Object.fromEntries(keys.map((key, index) => [key, { label: key.replaceAll('_', ' '),
        color: `var(--chart-${((slots[index] ?? index) % 8) + 1})` }]));
      const stacked = canvas.dataset.chartStacked === 'true';
      basecoat.chart(canvas, {
        type: canvas.dataset.chartType || 'bar', data, series, legend: canvas.dataset.chartLegend === 'true',
        options: { datasets: { bar: { maxBarThickness: 48 } },
                   scales: { x: { stacked }, y: { stacked, ticks: { display: true } } } },
      });
    });
  };
  // The New session action bar repeats the reviewed rate and guards, so the cost stays in view while editing.
  // The review carries every offered GPU class's fit for the current model and settings. Classes that cannot run it
  // are greyed out; a class the operator already chose stays selected and is flagged rather than silently swapped.
  const gpuFit = () => {
    try { return JSON.parse(document.getElementById('gpu-fit')?.dataset?.fit || 'null'); } catch (_) { return null; }
  };
  const blockedGpu = () => {
    const fits = gpuFit();
    const input = fits && document.querySelector('#new-form input[name="gpu"]:checked');
    const fit = input && fits[input.value];
    return fit?.status === 'blocked' ? { gpu: input.value, ...fit } : null;
  };
  const applyGpuFit = () => {
    const fits = gpuFit();
    const form = document.getElementById('new-form');
    if (!fits || !form) return;
    form.querySelectorAll('.cb-gpu').forEach((card) => {
      const input = card.querySelector('input[name="gpu"]');
      const fit = fits[input.value];
      if (!fit) return;
      card.dataset.fit = fit.status;
      card.title = fit.message;
      card.querySelector('.cb-gpu-fit').textContent = fit.label;
      const speed = card.querySelector('.cb-gpu-speed');
      if (speed) speed.textContent = fit.measured ? `${fit.measured.gen} tok/s measured` : '';
      input.disabled = fit.status === 'blocked' && !input.checked;
    });
    const note = document.getElementById('gpu-note');
    const blocked = blockedGpu();
    note.hidden = !blocked;
    if (!blocked) return;
    const alternative = document.getElementById('gpu-fit').dataset.suggest;
    const text = document.createElement('span');
    text.textContent = `${blocked.gpu} cannot run this model with these settings (${blocked.label}).`;
    note.replaceChildren(text);
    if (alternative) {
      const use = document.createElement('button');
      use.type = 'button';
      use.className = 'btn';
      use.dataset.size = 'sm';
      use.dataset.variant = 'outline';
      use.textContent = `Use ${alternative} instead`;
      use.addEventListener('click', () => {
        const radio = form.querySelector(`input[name="gpu"][value="${alternative}"]`);
        radio.disabled = false;
        radio.checked = true;
        radio.dispatchEvent(new Event('change', { bubbles: true }));
        radio.focus();
      });
      note.append(use);
    }
  };
  const updateCreateSummary = () => {
    const lines = document.getElementById('cost-lines');
    const target = document.getElementById('create-bar-summary');
    if (lines?.dataset?.summary && target) target.textContent = lines.dataset.summary.split('. ')[0];
    const blocked = blockedGpu();
    if (blocked && target) target.textContent = `${blocked.gpu} cannot run this model. Choose another GPU.`;
    target?.classList?.toggle('cb-blocked', Boolean(blocked));
  };
  document.addEventListener('htmx:afterSwap', (event) => {
    if (event.target.isConnected) renderCharts(event.target);
    if (event.target.id === 'estimate') { applyGpuFit(); updateCreateSummary(); }
  });
  const updateFilterSummary = (filter) => {
    const count = filter.querySelectorAll('input[type="checkbox"]:checked').length;
    const label = filter.querySelector('[data-filter-count]');
    if (label) label.textContent = count ? `${count} selected` : 'All';
    filter.dataset.active = String(count > 0);
  };
  document.addEventListener('change', (event) => {
    const filter = event.target.closest?.('.cb-query-filters .cb-filter');
    if (filter) updateFilterSummary(filter);
  });
  let logsPaused = false;
  window.cbToggleLogs = (button) => {
    logsPaused = !logsPaused;
    button.textContent = logsPaused ? 'Resume' : 'Pause';
    button.setAttribute('aria-pressed', String(logsPaused));
    htmx.trigger(document.getElementById('session-logs'), logsPaused ? 'htmx:abort' : 'refresh');
  };
  document.addEventListener('htmx:beforeRequest', (event) => {
    if (event.detail.elt?.id === 'session-logs' && logsPaused) event.preventDefault();
  });
  document.addEventListener('htmx:beforeOnLoad', (event) => {
    if (event.detail.elt?.id === 'session-logs' && logsPaused) event.preventDefault();
  });
  document.addEventListener('htmx:afterRequest', (event) => {
    const target = event.detail.elt;
    if (target?.id !== 'session-logs' || !target.isConnected || !event.detail.successful || logsPaused) return;
    const request = event.detail.xhr;
    const url = new URL(target.getAttribute('hx-get'), location.origin);
    url.searchParams.set('after', request.getResponseHeader('X-Log-Cursor') || '0');
    url.searchParams.set('after_event', request.getResponseHeader('X-Event-Cursor') || '0');
    target.setAttribute('hx-get', url.pathname + url.search);
    htmx.process(target);
    const log = document.getElementById('session-log-lines');
    const feed = document.getElementById('session-log-events');
    if (log) {
      const lines = log.textContent.split('\n');
      if (lines.at(-1) === '') lines.pop();
      if (lines.length > 500) log.textContent = lines.slice(-500).join('\n') + '\n';
    }
    if (feed) while (feed.children.length > 200) feed.firstElementChild.remove();
    if (document.getElementById('logs-autoscroll')?.checked) {
      for (const panel of [log, feed]) if (panel) panel.scrollTop = panel.scrollHeight;
    }
  });

  window.cbObserveSession = async () => {
    const session = document.getElementById('session-watch');
    const state = document.getElementById('session-controls')?.dataset.state;
    if (!session || document.visibilityState !== 'visible' || ['stopped', 'failed', 'stopping'].includes(state)) return;
    try {
      await fetch(`/ui/sessions/${session.dataset.sessionId}/watch`, { method: 'POST',
        headers: { 'X-CSRF-Token': csrf() } });
    } catch (_) {}
  };
  document.addEventListener('visibilitychange', cbObserveSession);

  window.cbPresetChanged = (select) => {
    const form = document.getElementById('new-form');
    const custom = select.value === 'custom';
    form.dataset.custom = String(custom);
    const engineControl = document.getElementById('engine');
    const engine = custom ? engineControl.value : select.selectedOptions[0]?.dataset.engine || 'llamacpp';
    if (form.dataset.engine && form.dataset.engine !== engine) {
      document.getElementById('kv').value = '';
      document.getElementById('extra_args').value = '';
      document.getElementById('refusal_projection').value = '';
    }
    form.dataset.engine = engine;
    engineControl.value = engine;
    form.querySelectorAll('.cb-custom').forEach((field) => {
      const enabled = custom && (!field.dataset.engineField || field.dataset.engineField.split(' ').includes(engine));
      field.hidden = !enabled;
      field.querySelectorAll('input, select').forEach((control) => control.disabled = !enabled);
    });
    document.getElementById('kv').setAttribute('list', `kv-types-${engine}`);
    document.getElementById('extra_args').placeholder = engine === 'exllamav3'
      ? 'model.chunk_size=4096' : engine === 'strata' ? '--ple-io direct' : '--reasoning-budget 4096';
    document.getElementById('projection-field').hidden = engine !== 'strata';
    document.getElementById('refusal_projection').disabled = engine !== 'strata';
    document.getElementById('ctx').placeholder = custom && engine === 'strata' ? '131072' : 'preset default';
    document.getElementById('parallel').placeholder = engine === 'strata' ? '1 (FIFO, single sequence)' : 'preset default';
    document.getElementById('kv').placeholder = custom && engine === 'strata' ? 'int8' : 'preset default';
    document.getElementById('repo').placeholder = engine === 'strata'
      ? 'ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF' : 'org/model';
    document.getElementById('file').placeholder = engine === 'strata'
      ? 'IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf' : 'model-Q4_K_M.gguf';
  };
  window.cbReconfigureChanged = (select) => {
    const engine = select.selectedOptions[0]?.dataset.engine || 'llamacpp';
    const projection = document.getElementById('rc-projection');
    document.getElementById('rc-projection-field').hidden = engine !== 'strata';
    projection.disabled = engine !== 'strata';
    if (engine !== 'strata') projection.value = '';
    document.getElementById('rc-kv').setAttribute('list', `rc-kv-types-${engine}`);
    document.getElementById('rc-extra').placeholder = engine === 'exllamav3'
      ? 'model.chunk_size=4096' : engine === 'strata' ? '--ple-io direct' : '--reasoning-budget 4096';
  };
  window.cbReviewCreate = async () => {
    const form = document.getElementById('new-form');
    if (!form.reportValidity()) return;
    const fields = new FormData(form);
    const reviewed = new URLSearchParams(fields).toString();
    try {
      const response = await fetch('/ui/new/estimate', { method: 'POST',
        headers: { 'X-CSRF-Token': csrf() }, body: fields });
      if (form.isConnected === false) return;
      if (!response.ok) {
        cbToast('error', 'Cannot review', (await response.json()).error || 'Check the session fields');
        return;
      }
      const markup = await response.text();
      if (form.isConnected === false) return;
      if (new URLSearchParams(new FormData(form)).toString() !== reviewed) {
        cbToast('info', 'Review again', 'The form changed while calculating cost');
        return;
      }
      const target = document.getElementById('estimate');
      target.innerHTML = markup;
      htmx.process(target);
      applyGpuFit();
      updateCreateSummary();
      const blocked = blockedGpu();
      if (blocked) {
        cbToast('error', `${blocked.gpu} cannot run this model`, `${blocked.message}. Choose a GPU that is not greyed out.`);
        return;
      }
      const lines = document.getElementById('cost-lines');
      document.getElementById('create-summary').textContent =
        `${lines ? lines.dataset.summary : ''}. This spends compute units from now until the session stops.`;
      document.getElementById('create-dialog').showModal();
    } catch (err) {
      if (form.isConnected !== false) cbToast('error', 'Cannot review', String(err));
    }
  };

  const fmt = (value, digits = 1) => (value === null || value === undefined || Number.isNaN(value) ? '-' : Number(value).toFixed(digits));
  const metricsLine = (metrics, target) => {
    target.textContent = [
      metrics.ttft !== undefined ? `TTFT ${fmt(metrics.ttft / 1000, 2)} s` : '',
      metrics.engineTps ? `engine ${fmt(metrics.engineTps)} t/s` : '',
      metrics.e2eTps ? `end-to-end ${fmt(metrics.e2eTps)} t/s` : '',
      metrics.prompt !== undefined ? `prompt ${metrics.prompt} tok` : '',
      metrics.out !== undefined ? `out ${metrics.out} tok` : '',
      `total ${fmt(metrics.total / 1000, 2)} s`, `HTTP ${metrics.status}`,
      'recorded as client colabinet-playground',
    ].filter(Boolean).join(' · ');
  };
  const numbers = (obj) => {
    const timings = obj.timings || {};
    const usage = obj.usage || {};
    return {
      prompt: usage.prompt_tokens ?? (timings.prompt_n !== undefined ? timings.prompt_n + (timings.cache_n || 0) : undefined),
      out: usage.completion_tokens ?? timings.predicted_n,
      engineTps: timings.predicted_per_second || (usage.completion_time ? usage.completion_tokens / usage.completion_time : undefined),
    };
  };
  window.cbPlaygroundExamples = () => {
    const python = document.getElementById('pg-python-example');
    const curl = document.getElementById('pg-curl-example');
    if (!python || !curl) return;
    const language = document.getElementById('pg-example-language').value;
    document.getElementById('pg-python-panel').hidden = language !== 'python';
    document.getElementById('pg-curl-panel').hidden = language !== 'curl';
    const copy = document.getElementById('pg-copy-example');
    copy.setAttribute('aria-label', `Copy ${language === 'python' ? 'Python' : 'curl'} example`);
    const session = document.getElementById('pg-session').selectedOptions[0];
    const limit = Number(document.getElementById('pg-max').value);
    const valid = session?.dataset.model && session?.dataset.endpoint && Number.isSafeInteger(limit) && limit > 0;
    copy.disabled = !valid;
    document.getElementById('pg-examples-error').hidden = Boolean(valid);
    if (!valid) {
      python.textContent = ''; curl.textContent = '';
      document.getElementById('pg-python-panel').hidden = true;
      document.getElementById('pg-curl-panel').hidden = true;
      return;
    }
    const thinking = document.getElementById('pg-thinking').value;
    const payload = {
      model: session.dataset.model, stream: false, max_tokens: limit,
      messages: [{ role: 'user', content: document.getElementById('pg-prompt').value }],
      chat_template_kwargs: { enable_thinking: thinking !== 'none' },
    };
    if (thinking !== 'none') {
      payload.reasoning_effort = thinking;
      payload.chat_template_kwargs.reasoning_effort = thinking;
    }
    const url = `${session.dataset.endpoint}/chat/completions`;
    const request = JSON.stringify(payload, null, 2);
    // JSON escapes keep arbitrary prompt text inside Python's raw string and the quoted shell heredoc.
    python.textContent = `import json
from urllib.request import Request, urlopen

payload = json.loads(r'''${request.replaceAll("'", '\\u0027')}''')
request = Request(
    ${JSON.stringify(url)},
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urlopen(request, timeout=120) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
`;
    const shellQuote = (value) => "'" + value.replaceAll("'", "'\\''") + "'";
    curl.textContent = `curl --fail --silent --show-error ${shellQuote(url)} \\
  --header 'Content-Type: application/json' \\
  --data-binary @- <<'COLABINET_REQUEST'
${request}
COLABINET_REQUEST
`;
  };
  for (const event of ['input', 'change']) document.addEventListener(event, (event) => {
    if (['pg-session', 'pg-prompt', 'pg-thinking', 'pg-max', 'pg-example-language'].includes(event.target.id)) cbPlaygroundExamples();
  });

  let playgroundController;
  let modelsController;
  window.cbPlaygroundSend = async () => {
    const root = document.querySelector('.cb-page');
    const element = (id) => root.querySelector(`#${id}`);
    const controller = new AbortController();
    playgroundController?.abort();
    playgroundController = controller;
    const stream = element('pg-stream').checked;
    const body = { session_id: Number(element('pg-session').value), prompt: element('pg-prompt').value, thinking: element('pg-thinking').value,
                   max_tokens: Number(element('pg-max').value), stream };
    element('pg-output').textContent = ''; element('pg-reasoning').textContent = ''; element('pg-reasoning-box').hidden = true;
    element('pg-hint').hidden = true;
    element('pg-metrics').textContent = 'Sending…'; element('pg-raw-out').hidden = !element('pg-raw').checked; element('pg-raw-out').textContent = '';
    element('pg-send').disabled = true;
    const t0 = performance.now();
    try {
      const response = await fetch('/ui/playground/send', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrf() }, body: JSON.stringify(body), signal: controller.signal });
      if (!stream || !response.headers.get('content-type')?.includes('text/event-stream')) {
        const data = await response.json();
        if (data.error) { cbToast('error', 'Not sent', data.error); element('pg-metrics').textContent = ''; return; }
        const msg = data.body?.choices?.[0]?.message || {};
        if (msg.reasoning_content) { element('pg-reasoning-box').hidden = false; element('pg-reasoning').textContent = msg.reasoning_content; }
        element('pg-output').textContent = msg.content ?? JSON.stringify(data.body?.error || data.body);
        const counts = numbers(data.body || {});
        metricsLine({ ...counts, total: data.total_ms, status: data.status, e2eTps: counts.out ? counts.out / (data.total_ms / 1000) : undefined }, element('pg-metrics'));
        element('pg-raw-out').textContent = JSON.stringify({ request: data.request, response: data.body }, null, 2);
        return;
      }
      const reader = response.body.getReader();
      const dec = new TextDecoder();
      let buf = '', ttft, last = {}, raw = '';
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        const text = dec.decode(value, { stream: true });
        buf = (buf + text).replace(/\r\n/g, '\n'); raw += text;
        let index;
        while ((index = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, index); buf = buf.slice(index + 2);
          const data = block.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim()).join('\n');
          if (!data || data === '[DONE]') continue;
          let ev; try { ev = JSON.parse(data); } catch (_) { continue; }
          const delta = ev.choices?.[0]?.delta || {};
          const piece = delta.content || '', think = delta.reasoning_content || delta.reasoning || '';
          if ((piece || think) && ttft === undefined) ttft = performance.now() - t0;
          if (think) { element('pg-reasoning-box').hidden = false; element('pg-reasoning').textContent += think; }
          if (piece) element('pg-output').textContent += piece;
          if (ev.timings || ev.usage) last = ev;
        }
      }
      const total = performance.now() - t0;
      const counts = numbers(last);
      metricsLine({ ...counts, ttft, total, status: response.status, e2eTps: counts.out && total > 0 ? counts.out / (total / 1000) : undefined }, element('pg-metrics'));
      element('pg-raw-out').textContent = `request:\n${JSON.stringify(body, null, 2)}\n\nresponse (SSE):\n${raw}`;
    } catch (err) {
      if (!controller.signal.aborted) cbToast('error', 'Request failed', String(err));
    } finally {
      if (playgroundController === controller) playgroundController = null;
      element('pg-send').disabled = false;
    }
  };
  window.cbPlaygroundModels = async () => {
    const target = document.getElementById('pg-raw-out');
    const controller = new AbortController();
    modelsController?.abort();
    modelsController = controller;
    try {
      const response = await fetch(`/ui/playground/models?session_id=${document.getElementById('pg-session').value}`,
        { signal: controller.signal });
      const data = await response.json();
      target.hidden = false;
      target.textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      if (!controller.signal.aborted) cbToast('error', 'Could not fetch models', String(err));
    } finally {
      if (modelsController === controller) modelsController = null;
    }
  };

  let observationTimer;
  const initializePage = () => {
    clearInterval(observationTimer);
    logsPaused = false;
    renderCharts(document);
    document.querySelectorAll('.cb-query-filters .cb-filter').forEach(updateFilterSummary);
    cbPlaygroundExamples();
    if (new URLSearchParams(location.search).has('reconfigure')) document.getElementById('reconfigure-dialog')?.showModal();
    const preset = document.getElementById('preset');
    if (preset) cbPresetChanged(preset);
    if (document.getElementById('session-watch')) {
      cbObserveSession();
      observationTimer = setInterval(cbObserveSession, 15000);
    }
  };
  document.addEventListener('cb:pageEnter', initializePage);
  document.addEventListener('cb:pageLeave', (event) => {
    clearInterval(observationTimer);
    playgroundController?.abort();
    modelsController?.abort();
    document.querySelectorAll('#cb-view [hx-get]').forEach((node) => {
      if (node !== event.detail?.source) htmx.trigger(node, 'htmx:abort');
    });
    document.querySelectorAll('#cb-view canvas[data-cb-chart]').forEach((canvas) => canvas._destroy?.());
  });
  document.addEventListener('DOMContentLoaded', () => {
    initializePage();
    pollToasts();
    setInterval(pollToasts, 3000);
  });
})();
