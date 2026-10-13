const fs = require('node:fs');
const { spawn } = require('node:child_process');
const [chrome, page, profile, script] = process.argv.slice(2);
const browser = spawn(chrome, ['--headless=new', '--disable-gpu', '--no-first-run',
  '--no-default-browser-check', '--disable-background-networking', '--remote-debugging-address=127.0.0.1',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank'], { stdio: 'ignore' });
const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
let socket;
const pending = new Map();

async function main() {
  const portFile = `${profile}/DevToolsActivePort`;
  let port;
  // Up to 30 s: a cold first start on a fresh CI machine took over 5 s.
  for (let attempt = 0; attempt < 600; attempt++) {
    if (fs.existsSync(portFile)) {
      const candidate = Number(fs.readFileSync(portFile, 'utf8').split('\n')[0]);
      if (candidate > 0 && candidate <= 65535) { port = candidate; break; }
    }
    await delay(50);
  }
  if (!port) throw Error('Browser debugging port did not become ready');
  const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  socket = new WebSocket(pages.find((entry) => entry.type === 'page').webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  let sequence = 0;
  socket.addEventListener('message', (event) => {
    const response = JSON.parse(event.data);
    const waiter = pending.get(response.id);
    if (!waiter) return;
    pending.delete(response.id);
    clearTimeout(waiter.timer);
    if (response.error) waiter.reject(Error(response.error.message)); else waiter.resolve(response.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => reject(Error(`Timed out: ${method}`)), 10000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const response = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (response.exceptionDetails) throw Error(response.exceptionDetails.text);
    return response.result.value;
  };
  await send('Emulation.setDeviceMetricsOverride', {
    width: 1440, height: 650, deviceScaleFactor: 1, mobile: false,
  });
  await send('Page.navigate', { url: page });
  let loaded = false;
  for (let attempt = 0; attempt < 100; attempt++) {
    loaded = await evaluate(`document.readyState === 'complete' && location.href === ${JSON.stringify(page)}`);
    if (loaded) break;
    await delay(50);
  }
  if (!loaded) throw Error('Browser page did not load');
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const task = new AsyncFunction('send', 'evaluate', fs.readFileSync(script, 'utf8'));
  process.stdout.write(JSON.stringify(await task(send, evaluate)));
}

main().catch((error) => { console.error(error); process.exitCode = 1; }).finally(async () => {
  for (const waiter of pending.values()) clearTimeout(waiter.timer);
  if (socket) socket.close();
  if (browser.exitCode === null) {
    const exited = new Promise((resolve) => browser.once('exit', resolve));
    browser.kill('SIGTERM');
    await exited;
  }
});
