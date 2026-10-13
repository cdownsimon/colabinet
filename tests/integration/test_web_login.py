"""Google sign-in through the real console routes and an isolated interactive CLI."""

import asyncio
import json
import os
import re
import shutil
import sys

import httpx
import pytest

from colabinet.api import create_app
from colabinet.config import Settings
from colabinet.db import Database
from colabinet.doctor import run as doctor
from colabinet.net import first_free, start_server, stop_server
from colabinet.server import build_supervisor
from colabinet.store import Store
from tests.fakes.browser import require_browser, run_browser

CSRF = {"X-CSRF-Token": "login-test-csrf"}
AUTH_URL = "https://accounts.google.com/o/oauth2/auth?state=fake-state&client_id=fake-client"


@pytest.fixture
async def login_console(tmp_path):
    binary = tmp_path / "colab"
    cached = tmp_path / "cached"
    journal = tmp_path / "calls.jsonl"
    pid = tmp_path / "login.pid"
    package = tmp_path / "colab_cli"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "auth.py").write_text(f"TOKEN_CONFIG_PATH = {str(cached)!r}\n")
    (package / "common.py").write_text('''
class State:
    _store = _settings_store = _history = None
state = State()
def setup_logging(log_to_stderr): pass
''')
    (package / "state.py").write_text('''
class StateStore:
    def __init__(self, path): self.path = path
class SettingsStore(StateStore): pass
''')
    (package / "history.py").write_text('''
class HistoryLogger:
    def __init__(self, path): self.log_dir = path
''')
    binary.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
from colab_cli import auth
args = sys.argv[1:]
with open({str(journal)!r}, 'a') as file:
    file.write(json.dumps(args) + '\\n')
command = args[-1]
cached = Path(auth.TOKEN_CONFIG_PATH)
if command == 'version':
    print('Version: 0.7.4')
elif command == 'whoami':
    print('Email:         ' + (cached.read_text() or 'reviewer@example.com'))
elif command == 'usage':
    if not cached.exists():
        Path({str(pid)!r}).write_text(str(os.getpid()))
        print('To authorize colab-cli, visit this URL in any browser:', file=sys.stderr, flush=True)
        print(os.environ.get('FAKE_LOGIN_URL', {AUTH_URL!r}), file=sys.stderr, flush=True)
        code = input('Enter the authorization code: ')
        if code not in ('fake-code', 'other-code') or os.environ.get('FAKE_LOGIN_FAIL'):
            print('private-code=secret-do-not-show access_token=secret-token', file=sys.stderr)
            sys.exit(1)
        cached.write_text('other@example.com' if code == 'other-code' else 'reviewer@example.com')
    balance = 25 if cached.read_text() == 'other@example.com' else 10
    print(f'Current balance: {{balance:.2f}} compute units\\nUsage rate: 0.00/hr\\nActive assignments: 0')
else:
    sys.exit(8)
''')
    binary.chmod(0o700)
    settings = Settings(data_dir=tmp_path / "data", colab_bin=str(binary), notify=False)
    db = Database(settings.db_path)
    db.init()
    sup = build_supervisor(settings, Store(db))
    app = create_app(sup, "login-test-token", 7860, csrf="login-test-csrf")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:7860") as client:
        try:
            yield app, client, cached, journal, pid, sup
        finally:
            if hasattr(app.state, "google_login"):
                await app.state.google_login.close()
            await sup.shutdown()
            db.close()


async def wait_state(client, state):
    for _ in range(100):
        response = await client.get("/ui/auth")
        if f'data-auth-state="{state}"' in response.text:
            return response.text
        await asyncio.sleep(0.02)
    pytest.fail(f"Google login did not reach {state}: {response.text}")


def flow_id(markup):
    return re.search(r'name="flow_id" value="([^"]+)"', markup)[1]


async def start(client):
    response = await client.post("/ui/actions/auth/start", headers=CSRF)
    assert response.status_code == 200
    return await wait_state(client, "waiting")


async def test_settings_exposes_google_sign_in_and_account_check(login_console):
    _, client, *_ = login_console
    response = await client.get("/settings")
    assert response.status_code == 200
    assert "Sign in with Google" in response.text
    assert 'id="google-account"' in response.text
    assert "/ui/actions/auth/check" in response.text


async def test_web_login_exchanges_code_and_verifies_account_without_allocating(login_console):
    _, client, cached, journal, _, sup = login_console
    markup = await start(client)
    assert AUTH_URL.replace("&", "&amp;") in markup
    assert 'rel="noopener noreferrer"' in markup
    response = await client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": "fake-code"},
                                 headers=CSRF)
    assert response.status_code == 200 and "fake-code" not in response.text
    markup = await wait_state(client, "connected")
    assert "reviewer@example.com" in markup and not cached.exists()
    assert "fake-state" not in markup and "authorization-code" not in markup
    assert "no-store" in response.headers["cache-control"]
    assert sup.usage.balance == 10
    calls = [json.loads(line) for line in journal.read_text().splitlines()]
    assert calls == [["--auth=oauth2", "usage"], ["--auth=oauth2", "whoami"]]


async def test_cached_account_is_checked_in_the_ui(login_console):
    _, client, cached, *_ = login_console
    cached.touch()
    response = await client.post("/ui/actions/auth/check", headers=CSRF)
    assert response.status_code == 200
    markup = await wait_state(client, "connected")
    assert "reviewer@example.com" in markup and "Open Google sign-in" not in markup


async def connect(client, code="fake-code"):
    markup = await start(client)
    response = await client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": code},
                                 headers=CSRF)
    assert response.status_code == 200
    return await wait_state(client, "connected")


async def test_two_accounts_switch_balances_and_logout_only_active_credentials(login_console):
    app, client, cached, _, _, sup = login_console
    await connect(client)
    first = app.state.google_login.snapshot()["active_id"]
    first_cli = sup.cli
    markup = await connect(client, "other-code")
    second = app.state.google_login.snapshot()["active_id"]
    second_cli = sup.cli
    assert first != second and "reviewer@example.com" in markup and "other@example.com" in markup
    assert "Add account" in markup and "Log out" in markup and "Switch" in markup
    assert sup.usage.balance == 25 and not cached.exists()
    assert first_cli.profile != second_cli.profile
    assert first_cli.profile.joinpath("token.json").exists()
    response = await client.post("/ui/actions/auth/switch", data={"account_id": first}, headers=CSRF)
    assert response.status_code == 200
    await wait_state(client, "connected")
    assert sup.usage.balance == 10
    response = await client.post("/ui/actions/auth/logout", data={"account_id": first}, headers=CSRF)
    assert response.status_code == 200
    assert app.state.google_login.snapshot()["active_id"] is None
    assert sup.usage is None and await sup.cli.usage() is None
    assert not first_cli.profile.joinpath("token.json").exists()
    assert second_cli.profile.joinpath("token.json").exists()
    assert (await client.post("/ui/actions/auth/check", headers=CSRF)).status_code == 200
    assert app.state.google_login.snapshot()["active_id"] is None
    response = await client.post("/ui/actions/auth/switch", data={"account_id": second}, headers=CSRF)
    assert response.status_code == 200 and sup.usage.balance == 25
    # Re-authenticating an existing email replaces its credentials without duplicating the account.
    await connect(client)
    snapshot = app.state.google_login.snapshot()
    assert snapshot["active_id"] == first and len(snapshot["accounts"]) == 2


async def test_logout_persists_and_does_not_delete_shared_cli_credentials(login_console):
    app, client, cached, _, _, sup = login_console
    cached.touch()
    await client.post("/ui/actions/auth/check", headers=CSRF)
    await wait_state(client, "connected")
    assert (await client.post("/ui/actions/auth/logout", data={"account_id": "default"},
                              headers=CSRF)).status_code == 200
    assert cached.exists()
    restored = build_supervisor(sup.s, sup.store)
    try:
        restored_app = create_app(restored, "test", 7860, csrf="test")
        assert restored_app.state.google_login.snapshot()["active_id"] is None
        assert await restored.cli.usage() is None
    finally:
        await restored.shutdown()


async def test_cancelled_add_account_preserves_current_account_and_credentials(login_console):
    app, client, *_ = login_console
    await connect(client)
    before = app.state.google_login.snapshot()
    markup = await start(client)
    assert (await client.post("/ui/actions/auth/cancel", data={"flow_id": flow_id(markup)},
                              headers=CSRF)).status_code == 200
    after = app.state.google_login.snapshot()
    assert after["active_id"] == before["active_id"] and after["accounts"] == before["accounts"]


@pytest.mark.parametrize("fields", [{"state": "ready"}, {"state": "stopped", "stop_unconfirmed": 1},
                                    {"state": "failed", "name_reserved": 1}])
async def test_account_changes_require_confirmed_stopped_sessions(login_console, fields):
    app, client, *_, sup = login_console
    await connect(client)
    account_id = app.state.google_login.snapshot()["active_id"]
    await sup.store.insert_session(colab_name="owned", account_id=account_id, gpu_requested="T4", proxy_port=9000,
                                   hidden_port=9001, vm_port=9002, state_since=0, created_at=0, rate_cu_h=1,
                                   rate_source="test", idle_min=30, **fields)
    for action in ("start", "logout", "switch"):
        data = {} if action == "start" else {"account_id": account_id}
        response = await client.post(f"/ui/actions/auth/{action}", data=data, headers=CSRF)
        assert response.status_code == 409 and "Stop" in response.json()["error"]
    assert app.state.google_login.snapshot()["active_id"] == account_id


async def test_switch_rejects_unknown_account_and_stale_logout(login_console):
    app, client, *_ = login_console
    await connect(client)
    first = app.state.google_login.snapshot()["active_id"]
    await connect(client, "other-code")
    for account_id in ("../../token", "unknown"):
        assert (await client.post("/ui/actions/auth/switch", data={"account_id": account_id},
                                  headers=CSRF)).status_code == 409
    assert (await client.post("/ui/actions/auth/logout", data={"account_id": first},
                              headers=CSRF)).status_code == 409


async def test_saved_active_profile_restores_cli_and_ssh_bridge_after_restart(login_console):
    app, client, *_, sup = login_console
    await connect(client)
    await connect(client, "other-code")
    current = app.state.google_login.snapshot()["active_id"]
    restored = build_supervisor(sup.s, sup.store)
    try:
        assert restored.account_id == current and restored.cli.profile == sup.cli.profile
        assert await restored.cli.identity() == "other@example.com"
        assert (await restored.cli.usage()).balance == 25
        assert str(restored.cli.profile) in restored.tunnels.command_prefix
        assert sup.accounts.root.stat().st_mode & 0o777 == 0o700
        assert sup.accounts.path.stat().st_mode & 0o777 == 0o600
        assert (sup.cli.profile / "token.json").stat().st_mode & 0o777 == 0o600
    finally:
        await restored.shutdown()


async def test_failed_add_does_not_replace_account_or_leave_credentials(login_console, monkeypatch):
    app, client, *_, sup = login_console
    await connect(client)
    before = app.state.google_login.snapshot()
    profiles = set(sup.accounts.root.iterdir())
    monkeypatch.setenv("FAKE_LOGIN_FAIL", "1")
    markup = await start(client)
    await client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": "other-code"}, headers=CSRF)
    await wait_state(client, "failed")
    after = app.state.google_login.snapshot()
    assert after["active_id"] == before["active_id"] and after["accounts"] == before["accounts"]
    assert set(sup.accounts.root.iterdir()) == profiles and not sup.account_change_pending


async def test_invalid_saved_credentials_cannot_switch_to_a_different_identity(login_console):
    app, client, *_, sup = login_console
    await connect(client)
    first = app.state.google_login.snapshot()["active_id"]
    first_profile = sup.cli.profile
    await connect(client, "other-code")
    second = app.state.google_login.snapshot()["active_id"]
    (first_profile / "token.json").write_text("someone-else@example.com")  # fake credential, no real token
    response = await client.post("/ui/actions/auth/switch", data={"account_id": first}, headers=CSRF)
    assert response.status_code == 409 and app.state.google_login.snapshot()["active_id"] == second
    assert sup.usage.balance == 25 and not sup.account_change_pending


async def test_logout_drains_background_usage_before_deleting_credentials(login_console, monkeypatch):
    app, client, *_, sup = login_console
    await connect(client)
    account_id = app.state.google_login.snapshot()["active_id"]
    started, release = asyncio.Event(), asyncio.Event()
    previous_usage = sup.usage

    async def delayed_usage():
        started.set()
        await release.wait()
        return previous_usage

    monkeypatch.setattr(sup.cli, "usage", delayed_usage)
    background = asyncio.create_task(sup.usage_once())
    await started.wait()
    logout = asyncio.create_task(client.post("/ui/actions/auth/logout", data={"account_id": account_id}, headers=CSRF))
    await asyncio.sleep(0.05)
    assert not logout.done()
    release.set()
    await background
    assert (await logout).status_code == 200 and sup.usage is None


async def test_missing_active_credentials_still_allows_console_to_start_and_sign_in(login_console):
    _, client, *_, sup = login_console
    await connect(client)
    (sup.cli.profile / "token.json").unlink()
    restored = build_supervisor(sup.s, sup.store)
    try:
        assert await restored.cli.usage() is None
        assert restored.account_id == sup.account_id
    finally:
        await restored.shutdown()


async def test_missing_credentials_with_billing_history_can_reauthenticate_same_owner(login_console):
    app, client, *_, sup = login_console
    await connect(client)
    owner, profile = sup.account_id, sup.cli.profile
    (profile / "sessions.json").write_text('{"owned": "private fake session metadata"}')
    await sup.store.insert_session(colab_name="owned", account_id=owner, gpu_requested="T4", proxy_port=9000,
                                   hidden_port=9001, vm_port=9002, state_since=0, created_at=0, rate_cu_h=1,
                                   rate_source="test", idle_min=30, state="stopped", stop_unconfirmed=1)
    (profile / "token.json").unlink()
    restored = build_supervisor(sup.s, sup.store)
    try:
        await restored.start()
        restored_app = create_app(restored, "test", 7860, csrf="login-test-csrf")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored_app),
                                     base_url="http://127.0.0.1:7860") as restored_client:
            response = await restored_client.post("/ui/actions/auth/start", data={"account_id": owner}, headers=CSRF)
            assert response.status_code == 200
            markup = await wait_state(restored_client, "waiting")
            await restored_client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": "other-code"},
                                       headers=CSRF)
            await wait_state(restored_client, "failed")
            assert restored.account_id == owner and not (profile / "token.json").exists()
            response = await restored_client.post("/ui/actions/auth/start", data={"account_id": owner}, headers=CSRF)
            assert response.status_code == 200
            markup = await wait_state(restored_client, "waiting")
            await restored_client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": "fake-code"},
                                       headers=CSRF)
            await wait_state(restored_client, "connected")
            assert restored.account_id == owner and restored.cli.profile == profile
            assert (profile / "sessions.json").read_text() == '{"owned": "private fake session metadata"}'
            assert await restored.cli.identity() == "reviewer@example.com"
    finally:
        await restored.shutdown()


async def test_doctor_checks_selected_web_account_instead_of_global_credentials(login_console):
    _, client, cached, *_, sup = login_console
    await connect(client)
    output = []
    await doctor(sup.s, out=output.append)
    assert any(line.startswith("OK Google account:") for line in output)
    assert not cached.exists()


async def test_profile_launcher_does_not_shadow_official_package(login_console, tmp_path, monkeypatch):
    _, _, _, _, _, sup = login_console
    library = tmp_path / "library"
    library.mkdir()
    shutil.move(tmp_path / "colab_cli", library)
    monkeypatch.setenv("PYTHONPATH", str(library))
    candidate = sup.accounts.candidate(sup.cli)
    assert await candidate.version() == "0.7.4"
    assert not (candidate.profile / "token.json").exists()
    sup.accounts.discard(candidate)


async def test_stale_external_stop_cannot_operate_on_new_account(login_console):
    app, client, _, journal, *_ = login_console
    await connect(client)
    first = app.state.google_login.snapshot()["active_id"]
    await connect(client, "other-code")
    before = journal.read_text()
    response = await client.post("/ui/actions/external/reused-name/stop",
                                 data={"confirm": "yes", "account_id": first}, headers=CSRF)
    assert response.status_code == 409 and "account changed" in response.json()["error"]
    assert journal.read_text() == before


async def test_failed_metadata_write_preserves_active_account_and_cleans_candidate(login_console, monkeypatch):
    app, client, *_, sup = login_console
    await connect(client)
    before = app.state.google_login.snapshot()
    paths = set(sup.accounts.root.iterdir())
    old_cli = sup.cli

    def denied():
        raise PermissionError("synthetic failure")

    monkeypatch.setattr(sup.accounts, "_save", denied)
    markup = await start(client)
    await client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": "other-code"}, headers=CSRF)
    await wait_state(client, "failed")
    after = app.state.google_login.snapshot()
    assert after["active_id"] == before["active_id"] and after["accounts"] == before["accounts"]
    assert sup.accounts.active == sup.account_id and sup.cli is old_cli
    assert set(sup.accounts.root.iterdir()) == paths


async def test_failed_logout_metadata_write_restores_credentials(login_console, monkeypatch):
    app, client, *_, sup = login_console
    await connect(client)
    owner, token = sup.account_id, sup.cli.profile / "token.json"
    previous = token.read_text()  # fixture credentials contain only a synthetic email

    def denied():
        raise PermissionError("synthetic failure")

    monkeypatch.setattr(sup.accounts, "_save", denied)
    response = await client.post("/ui/actions/auth/logout", data={"account_id": owner}, headers=CSRF)
    assert response.status_code == 409
    assert sup.accounts.active == sup.account_id == owner and token.read_text() == previous
    assert app.state.google_login.snapshot()["accounts"][0]["signed_in"]


async def test_failed_reauthentication_token_replace_restores_previous_credentials(login_console, monkeypatch):
    _, client, *_, sup = login_console
    await connect(client)
    token = sup.cli.profile / "token.json"
    previous = token.read_text()
    replace = os.replace

    def denied(source, destination):
        if str(source).endswith(".tmp") and destination == token:
            raise PermissionError("synthetic replace failure")
        return replace(source, destination)

    monkeypatch.setattr(os, "replace", denied)
    await client.post("/ui/actions/auth/start", data={"account_id": sup.account_id}, headers=CSRF)
    markup = await wait_state(client, "waiting")
    await client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": "fake-code"}, headers=CSRF)
    await wait_state(client, "failed")
    assert token.read_text() == previous
    assert not list(sup.cli.profile.glob("*.backup"))


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                    reason="root ignores the read-only directory this test uses to make cleanup fail")
async def test_candidate_cleanup_failure_cannot_leave_account_change_pending(login_console):
    app, client, *_, sup = login_console
    await connect(client)
    owner = sup.account_id
    markup = await start(client)
    sup.accounts.root.chmod(0o500)
    try:
        await client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": "other-code"},
                          headers=CSRF)
        await wait_state(client, "failed")
    finally:
        sup.accounts.root.chmod(0o700)
    assert sup.account_id == sup.accounts.active == owner and not sup.account_change_pending
    assert app.state.google_login._task.exception() is None
    assert await sup.cli.identity() == "reviewer@example.com"


async def test_cancelled_recovery_is_resumed_after_account_check(login_console, monkeypatch):
    _, client, *_, sup = login_console
    await connect(client)
    owner = sup.account_id
    sid = await sup.store.insert_session(colab_name="owned", account_id=owner, gpu_requested="T4", proxy_port=9000,
                                         hidden_port=9001, vm_port=9002, state_since=0, created_at=0, rate_cu_h=1,
                                         rate_source="test", idle_min=30, state="ready")
    (sup.cli.profile / "token.json").unlink()
    restored = build_supervisor(sup.s, sup.store)
    entered, release = asyncio.Event(), asyncio.Event()
    drivers = []

    async def delayed_bind(row):
        entered.set()
        await release.wait()

    monkeypatch.setattr(restored, "_bind_proxy", delayed_bind)
    monkeypatch.setattr(restored, "_start_driver", drivers.append)
    restored_app = create_app(restored, "test", 7860, csrf="login-test-csrf")
    login = restored_app.state.google_login
    try:
        await restored.start()
        await login.start(owner)
        for _ in range(100):
            if login.state == "waiting":
                break
            await asyncio.sleep(0.02)
        await login.submit(login.flow_id, "fake-code")
        await asyncio.wait_for(entered.wait(), 3)
        await login.cancel(login.flow_id)
        assert sid in restored._paused_for_account and restored.cli.enabled
        release.set()
        await login.check()
        await asyncio.wait_for(login._task, 3)
        assert login.state == "connected" and sid not in restored._paused_for_account
        assert drivers == [sid]
    finally:
        release.set()
        await login.close()
        await restored.shutdown()


async def test_stop_during_recovery_cannot_leave_a_rebound_proxy(login_console, monkeypatch):
    _, client, *_, sup = login_console
    await connect(client)
    sid = await sup.store.insert_session(colab_name="owned", account_id=sup.account_id, gpu_requested="T4",
                                         proxy_port=9000, hidden_port=9001, vm_port=9002, state_since=0, created_at=0,
                                         rate_cu_h=1, rate_source="test", idle_min=30, state="ready")
    sup._paused_for_account.add(sid)
    entered, release = asyncio.Event(), asyncio.Event()
    bindings = set()

    async def delayed_bind(session_id, port, app):
        entered.set()
        await release.wait()
        bindings.add(session_id)

    async def unbind(session_id):
        bindings.discard(session_id)

    async def confirmed_stop(session_id, foreign=False):
        await sup._finish(session_id, "stopped", "user", vm_created=True)

    monkeypatch.setattr(sup.proxies, "bind", delayed_bind)
    monkeypatch.setattr(sup.proxies, "unbind", unbind)
    monkeypatch.setattr(sup, "_start_driver", lambda session_id: None)
    monkeypatch.setattr(sup, "_stop_flow", confirmed_stop)
    recovery = asyncio.create_task(sup.resume_account_sessions())
    await asyncio.wait_for(entered.wait(), 3)
    stop = asyncio.create_task(sup.stop(sid))
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.gather(recovery, stop)
    assert (await sup.store.session(sid))["state"] == "stopped" and not bindings


async def test_duplicate_start_and_stale_code_cannot_replace_active_login(login_console):
    _, client, _, journal, *_ = login_console
    markup = await start(client)
    response = await client.post("/ui/actions/auth/start", headers=CSRF)
    assert response.status_code == 200 and flow_id(response.text) == flow_id(markup)
    response = await client.post("/ui/actions/auth/code", data={"flow_id": "stale", "code": "fake-code"}, headers=CSRF)
    assert response.status_code == 409
    assert len(journal.read_text().splitlines()) == 1


async def test_cancel_terminates_only_the_owned_login_process(login_console):
    _, client, _, _, pid, *_ = login_console
    markup = await start(client)
    process_id = int(pid.read_text())
    response = await client.post("/ui/actions/auth/cancel", data={"flow_id": flow_id(markup)}, headers=CSRF)
    assert response.status_code == 200
    await wait_state(client, "cancelled")
    with pytest.raises(ProcessLookupError):
        os.kill(process_id, 0)


async def test_timeout_cleans_up_and_offers_retry(login_console):
    app, client, _, _, pid, *_ = login_console
    app.state.google_login.timeout = 1
    response = await client.post("/ui/actions/auth/start", headers=CSRF)
    assert response.status_code == 200
    markup = await wait_state(client, "failed")
    assert "expired" in markup and "Sign in with Google" in markup
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


@pytest.mark.parametrize("code", ["", "fake-code\nother-command", "x" * 4097])
async def test_invalid_code_is_rejected_without_consuming_the_flow(login_console, code):
    _, client, *_ = login_console
    markup = await start(client)
    response = await client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": code}, headers=CSRF)
    assert response.status_code == 409
    assert 'data-auth-state="waiting"' in (await client.get("/ui/auth")).text


async def test_upstream_failure_does_not_echo_cli_secrets(login_console, monkeypatch):
    _, client, *_ = login_console
    monkeypatch.setenv("FAKE_LOGIN_FAIL", "1")
    markup = await start(client)
    await client.post("/ui/actions/auth/code", data={"flow_id": flow_id(markup), "code": "fake-code"}, headers=CSRF)
    markup = await wait_state(client, "failed")
    assert "secret-do-not-show" not in markup and "secret-token" not in markup
    assert "Sign in with Google" in markup


@pytest.mark.parametrize("url", ["https://evil.example/oauth", "https://accounts.google.com.evil.example/oauth",
                                  "http://accounts.google.com/o/oauth2/auth"])
async def test_only_google_authorization_urls_are_rendered(login_console, monkeypatch, url):
    _, client, *_ = login_console
    monkeypatch.setenv("FAKE_LOGIN_URL", url)
    response = await client.post("/ui/actions/auth/start", headers=CSRF)
    assert response.status_code == 200
    markup = await wait_state(client, "failed")
    assert url not in markup and "Open Google sign-in" not in markup


@pytest.mark.parametrize("action", ["start", "check", "code", "cancel", "switch", "logout"])
async def test_auth_actions_require_csrf_and_local_origin(login_console, action):
    _, client, *_ = login_console
    path = f"/ui/actions/auth/{action}"
    assert (await client.post(path)).status_code == 403
    assert (await client.post(path, headers={**CSRF, "Origin": "https://evil.example"})).status_code == 403
    assert (await client.post(path, headers={**CSRF, "Host": "evil.example:7860"})).status_code == 403


async def test_adc_mode_explains_account_setup_and_prevents_oauth_login(login_console):
    app, client, *_, sup = login_console
    sup.s.colab_auth = "adc"
    app.state.google_login.mode = "adc"
    response = await client.get("/settings")
    assert "gcloud auth application-default login" in response.text
    assert "Sign in with Google" not in response.text
    assert (await client.post("/ui/actions/auth/start", headers=CSRF)).status_code == 409


async def test_server_shutdown_cleans_up_pending_login(login_console):
    app, client, _, _, pid, *_ = login_console
    server, task = await start_server(app, first_free(52000, 52999), lifespan=True)
    try:
        await start(client)
        process_id = int(pid.read_text())
    finally:
        await stop_server(server, task)
    with pytest.raises(ProcessLookupError):
        os.kill(process_id, 0)


async def test_new_sign_in_waits_for_previous_cancellation_to_finish(login_console, monkeypatch):
    app, client, *_ = login_console
    markup = await start(client)
    login = app.state.google_login
    cleaned, release = asyncio.Event(), asyncio.Event()
    close = login.close

    async def delayed_cleanup():
        await close()
        cleaned.set()
        await release.wait()

    monkeypatch.setattr(login, "close", delayed_cleanup)
    cancellation = asyncio.create_task(client.post("/ui/actions/auth/cancel",
                                                   data={"flow_id": flow_id(markup)}, headers=CSRF))
    await asyncio.wait_for(cleaned.wait(), 3)
    replacement = asyncio.create_task(client.post("/ui/actions/auth/start", headers=CSRF))
    await asyncio.sleep(0.05)
    release.set()
    old, new = await asyncio.gather(cancellation, replacement)
    assert old.status_code == new.status_code == 200
    new_markup = await wait_state(client, "waiting")
    assert flow_id(new_markup) != flow_id(markup)


@pytest.mark.slow
async def test_browser_completes_sign_in_and_keeps_code_during_status_refresh(login_console, tmp_path):
    require_browser()
    _, _, _, journal, _, sup = login_console
    port = first_free(51000, 51999)
    app = create_app(sup, "login-test-token", port, csrf="login-test-csrf")
    server, task = await start_server(app, port, lifespan=True)
    directory = tmp_path / "browser"
    directory.mkdir()
    try:
        result = await asyncio.to_thread(run_browser, directory, f"http://127.0.0.1:{port}/settings", r'''
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function wait(expression) {
  for (let attempt = 0; attempt < 200; attempt++) {
    if (await evaluate(expression)) { await delay(50); return; }
    await delay(30);
  }
  throw Error(`Browser condition did not become true: ${expression}`);
}
await wait(`!!document.getElementById('google-login-start')`);
await evaluate(`document.getElementById('google-login-start').click(); true`);
await wait(`document.getElementById('google-account').dataset.authState === 'waiting'`);
await evaluate(`const input = document.getElementById('google-authorization-code');
  input.value = 'fake-'; input.focus(); true`);
await delay(1400);
const retained = await evaluate(`document.activeElement === document.getElementById('google-authorization-code') &&
  document.getElementById('google-authorization-code').value === 'fake-'`);
await evaluate(`document.getElementById('google-authorization-code').value = 'fake-code';
  document.getElementById('google-code-form').requestSubmit(); true`);
await wait(`document.getElementById('google-account').dataset.authState === 'connected'`);
await evaluate(`document.getElementById('google-add-account').click(); true`);
await wait(`document.getElementById('google-account').dataset.authState === 'waiting'`);
await evaluate(`document.getElementById('google-authorization-code').value = 'other-code';
  document.getElementById('google-code-form').requestSubmit(); true`);
await wait(`document.getElementById('google-account').dataset.authState === 'connected' &&
  document.getElementById('google-account-status').textContent.includes('other@example.com')`);
await evaluate(`document.querySelector('[data-account-email="reviewer@example.com"] button').click(); true`);
await wait(`document.getElementById('google-account-status').textContent.includes('reviewer@example.com')`);
const switched = await evaluate(`document.querySelector('[data-account-email="reviewer@example.com"]').textContent
  .includes('Current account')`);
await evaluate(`document.getElementById('google-account-logout').click(); true`);
await wait(`document.getElementById('google-account').dataset.authState === 'signed_out'`);
const loggedOut = await evaluate(`document.querySelector('[data-account-email="reviewer@example.com"]').textContent
  .includes('Logged out') && !!document.querySelector('[data-account-email="other@example.com"] button')`);
await evaluate(`document.querySelector('[data-account-email="other@example.com"] button').click(); true`);
await wait(`document.getElementById('google-account-status').textContent.includes('other@example.com')`);
await evaluate(`document.querySelector('[data-account-email="reviewer@example.com"] button').click(); true`);
await wait(`document.getElementById('google-account').dataset.authState === 'waiting'`);
await evaluate(`document.getElementById('google-authorization-code').value = 'fake-code';
  document.getElementById('google-code-form').requestSubmit(); true`);
await wait(`document.getElementById('google-account').dataset.authState === 'connected' &&
  document.getElementById('google-account-status').textContent.includes('reviewer@example.com')`);
return await evaluate(`({retained: ${retained}, connected: document.getElementById('google-account').textContent
  .includes('reviewer@example.com'), codeRemoved: !document.getElementById('google-authorization-code'),
  switched: ${switched}, loggedOut: ${loggedOut}, accounts: document.querySelectorAll('[data-account-email]').length,
  privateStored: [localStorage, sessionStorage].some((storage) => Object.values(storage).some((value) =>
    value.includes('fake-code') || value.includes('fake-state'))),
  consentRemoved: !document.getElementById('google-consent-link')})`);
''')
        assert result == {"retained": True, "connected": True, "codeRemoved": True, "privateStored": False,
                          "switched": True, "loggedOut": True, "accounts": 2,
                          "consentRemoved": True}
        assert sup.usage.balance == 10
        assert all(json.loads(line)[-1] in ("usage", "whoami") for line in journal.read_text().splitlines())
    finally:
        await stop_server(server, task)
