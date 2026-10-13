import json

import httpx
import pytest

from colabinet.api import create_app
from colabinet.db import Database
from colabinet.store import Store
from colabinet.web import views
from tests.integration.test_console import CSRF

pytestmark = pytest.mark.slow

TRACEBACK = ('Traceback (most recent call last):\n' + '  backend context\n' * 50
             + 'Authorization: Bearer diagnostic-secret\n'
             + 'https://example.test/?colab-runtime-proxy-token=proxy-secret\n'
             + 'HTTP error: ' + json.dumps({'error': json.dumps({'access_token': 'nested-token-secret'})}) + '\n'
             + '-----BEGIN PRIVATE KEY-----\nprivate-key-secret\n-----END PRIVATE KEY-----\n'
             + '<script>alert("bad")</script>\n'
             + 'QuotaError: accelerator A100 unavailable')


async def test_failed_creation_has_durable_redacted_output_and_failed_run(world):
    world.fake.fault(new="not_available", new_error=TRACEBACK)
    sid = await world.sup.create(world.request(gpu="A100"))
    row = await world.wait_state(sid, "failed")
    run = await world.store.current_run(sid)
    assert run["status"] == "failed"
    assert 'QuotaError' in run['error']
    assert row['new_exit'] == 1
    assert not await world.store.name_in_use('llm')
    other_db = Database(world.settings.db_path)
    other_db.init()
    try:
        diagnostic = await Store(other_db).creation_diagnostics(sid)
        assert diagnostic['exit_code'] == 1
        assert 'Traceback (most recent call last):' in diagnostic['output']
        assert 'QuotaError: accelerator A100 unavailable' in diagnostic['error']
        assert 'diagnostic-secret' not in diagnostic['output']
        assert 'proxy-secret' not in diagnostic['output']
        assert 'nested-token-secret' not in diagnostic['output']
        assert 'private-key-secret' not in diagnostic['output']
    finally:
        other_db.close()


async def test_failed_creation_is_visible_in_console_and_download(world):
    world.fake.fault(new="not_available", new_error=TRACEBACK)
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, 'failed')
    app = create_app(world.sup, 'api-test-token', 7860, csrf=CSRF)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:7860') as client:
        for path in (f'/sessions/{sid}', f'/ui/sessions/{sid}/status'):
            response = await client.get(path)
            assert 'Session creation failed' in response.text
            assert 'QuotaError: accelerator A100 unavailable' in response.text
            assert f'/sessions/{sid}?tab=logs' in response.text
            assert '<script>alert' not in response.text
        for path in ('/ui/overview/sessions', '/ui/sessions/table?tab=all'):
            response = await client.get(path)
            assert response.status_code == 200
            assert 'View error' in response.text
            assert 'QuotaError: accelerator A100 unavailable' in response.text
            assert f'/sessions/{sid}' in response.text
        response = await client.get(f'/sessions/{sid}?tab=logs')
        assert 'Creation log' in response.text
        assert 'Traceback (most recent call last):' in response.text
        assert 'Exit code 1' in response.text
        assert '<script>alert' not in response.text
        download = await client.get(f'/ui/sessions/{sid}/logs.txt')
        assert 'Traceback (most recent call last):' in download.text
        assert 'QuotaError' in download.text
        assert '<script>alert' in download.text  # plain text attachment
        assert 'diagnostic-secret' not in download.text
        assert 'proxy-secret' not in download.text
        assert 'nested-token-secret' not in download.text
        assert 'private-key-secret' not in download.text
        assert 'attachment' in download.headers['content-disposition']
        response = await client.get(f'/ui/sessions/{sid}/logs')
        assert 'Traceback (most recent call last):' in response.text
        assert 'diagnostic-secret' not in response.text
        cursor, event_cursor = response.headers['x-log-cursor'], response.headers['x-event-cursor']
        response = await client.get(f'/ui/sessions/{sid}/logs?after={cursor}&after_event={event_cursor}')
        assert 'beforeend:#session-log-events' not in response.text
        assert response.text.count('Traceback (most recent call last):') == 1
        assert 'hx-swap-oob="innerHTML"' in response.text  # replaces, rather than appends, creation output


async def test_creation_launch_error_is_inspectable_and_releases_name(world, monkeypatch):
    async def failed_spawn(*_):
        raise OSError('Executable could not start: access_token=launch-secret')

    monkeypatch.setattr(world.sup.cli, 'spawn_new', failed_spawn)
    sid = await world.sup.create(world.request())
    row = await world.wait_state(sid, 'failed')
    assert not row['name_reserved']
    assert not await world.store.name_in_use('llm')
    assert row['stopped_at'] == row['created_at']
    diagnostics = await world.store.creation_diagnostics(sid)
    assert 'Executable could not start' in diagnostics['error']
    assert 'launch-secret' not in diagnostics['output']
    assert (await world.store.current_run(sid))['status'] == 'failed'
    assert row['proxy_port'] not in world.sup.proxies.ports()


async def test_old_failure_message_is_visible_when_creation_output_was_not_retained(world):
    world.fake.fault(new='not_available')
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, 'failed')
    await world.store.update_session(sid, create_log=None, create_error=None)
    await world.store.event(world.sup.now(), 'finish_message', session_id=sid,
                            message='LegacyError: missing credentials access_token=old-secret')
    await world.store.update_run((await world.store.current_run(sid))['id'], status='pending', error=None)
    diagnostic = await world.store.creation_diagnostics(sid)
    assert 'LegacyError: missing credentials' in diagnostic['error']
    assert 'old-secret' not in diagnostic['error']
    app = create_app(world.sup, 'api-test-token', 7860, csrf=CSRF)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:7860') as client:
        response = await client.get(f'/sessions/{sid}')
        assert 'LegacyError: missing credentials' in response.text
        assert 'old-secret' not in response.text
        download = await client.get(f'/ui/sessions/{sid}/logs.txt')
        assert 'old-secret' not in download.text


async def test_updated_templates_work_with_an_already_running_console(world, monkeypatch):
    world.fake.fault(new='not_available')
    sid = await world.sup.create(world.request())
    await world.wait_state(sid, 'failed')
    current_detail = views.session_detail

    async def old_detail(*args):
        detail = await current_detail(*args)
        detail.pop('creation')
        detail['s'].pop('creation_error')
        return detail

    monkeypatch.setattr(views, 'session_detail', old_detail)
    app = create_app(world.sup, 'api-test-token', 7860, csrf=CSRF)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://127.0.0.1:7860') as client:
        for path in (f'/sessions/{sid}', f'/sessions/{sid}?tab=logs', f'/ui/sessions/{sid}/status'):
            assert (await client.get(path)).status_code == 200
