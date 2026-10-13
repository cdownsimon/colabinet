(() => {
  let activeRequest;
  let activePage;
  let currentPath = location.pathname + location.search;
  const view = () => document.getElementById('cb-view');
  const loading = (request) => {
    activeRequest = request;
    document.getElementById('cb-loading').hidden = !request;
    if (request) view().setAttribute('aria-busy', 'true'); else view().removeAttribute('aria-busy');
  };
  const startRequest = (request) => {
    const previous = activeRequest;
    loading(request);
    if (previous && previous !== request) previous.abort();
  };
  const parsePage = (markup) => {
    const response = new DOMParser().parseFromString(markup, 'text/html');
    return response.querySelector('#cb-view .cb-page[data-page]') ? response : null;
  };
  const historyError = (request) => {
    if (request !== activeRequest) return;
    loading(null);
    history.replaceState(history.state, '', currentPath);
    cbToast('error', 'Could not restore page', 'Try opening the page again.');
  };
  const enter = () => {
    const page = view().querySelector('[data-page]');
    if (!page || page === activePage) return;
    activePage = page;
    currentPath = location.pathname + location.search;
    const path = page.dataset.page === 'overview' ? '/' : `/${page.dataset.page}`;
    document.querySelectorAll('#sidebar .scrollbar a').forEach((link) => {
      if (link.getAttribute('href') === path) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    const badge = document.getElementById('nav-live-count');
    badge.textContent = page.dataset.liveCount;
    badge.hidden = Number(page.dataset.liveCount) === 0;
    document.dispatchEvent(new CustomEvent('cb:pageEnter'));
    const heading = view().querySelector('.cb-crumbs h1');
    if (heading) { heading.setAttribute('tabindex', '-1'); heading.focus({ preventScroll: true }); }
  };
  window.cbNavigate = (path) => htmx.ajax('GET', path, {
    source: view(), target: view(), select: '#cb-view > *', selectOOB: '#cmdk:outerHTML',
    swap: 'innerHTML show:top', push: 'true',
  }).catch(() => {});
  document.addEventListener('htmx:configRequest', (event) => {
    if (event.detail.target === view() && event.detail.verb === 'get') event.detail.parameters = {};
  });
  document.addEventListener('htmx:beforeCleanupElement', (event) => {
    const source = event.detail.elt;
    if (source.nodeType === Node.ELEMENT_NODE && source.matches('[hx-post]')) {
      source.addEventListener('htmx:beforeOnLoad', (response) => response.preventDefault(), { once: true });
    }
  });
  document.addEventListener('htmx:beforeRequest', (event) => {
    if (event.detail.target !== view() || event.detail.requestConfig.verb !== 'get') return;
    document.getElementById('cmdk')?.close();
    startRequest(event.detail.xhr);
  });
  document.addEventListener('htmx:beforeSwap', (event) => {
    if (event.detail.target !== view() || !event.detail.shouldSwap) return;
    if (!parsePage(event.detail.serverResponse)) {
      event.detail.shouldSwap = false;
      cbToast('error', 'Could not open page', 'The server did not return a console page.');
      return;
    }
    document.dispatchEvent(new CustomEvent('cb:pageLeave', { detail: { source: event.detail.requestConfig.elt } }));
  });
  document.addEventListener('htmx:afterSettle', (event) => { if (event.target === view()) enter(); });
  document.addEventListener('htmx:afterRequest', (event) => {
    if (event.detail.xhr === activeRequest) loading(null);
  });
  document.addEventListener('htmx:historyCacheMiss', (event) => {
    const request = event.detail.xhr;
    document.getElementById('cmdk')?.close();
    startRequest(request);
    const onLoad = request.onload;
    request.onload = function (loadEvent) {
      if (request !== activeRequest) return;
      if (request.status >= 200 && request.status < 400 && !parsePage(request.responseText)) {
        historyError(request);
        return;
      }
      onLoad.call(this, loadEvent);
    };
    ['error', 'abort', 'timeout'].forEach((name) => {
      request.addEventListener(name, () => historyError(request), { once: true });
    });
  });
  document.addEventListener('htmx:historyCacheMissLoad', (event) => {
    document.dispatchEvent(new CustomEvent('cb:pageLeave'));
    const menu = parsePage(event.detail.response).getElementById('cmdk');
    if (menu) htmx.swap('#cmdk', menu.outerHTML, { swapStyle: 'outerHTML', settleDelay: 0 });
  });
  document.addEventListener('htmx:historyRestore', () => { loading(null); enter(); });
  document.addEventListener('htmx:historyCacheMissLoadError', (event) => historyError(event.detail.xhr));
})();
