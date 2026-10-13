"""The sessions table must not claim a billing state it cannot see.

Both cases below are reachable from the console's own search box, so they are asserted
through the real endpoint rather than by rendering the template directly:

- filtering used the raw query while displaying the stripped one, so a whitespace-only
  search emptied a list that was not empty;
- the empty-state copy asserted that sessions were still running, which was wrong on an
  empty database and wrong on tabs whose sessions are stopped rather than billing.
"""

import httpx
import pytest

from colabinet.api import create_app
from tests.integration.conftest import make_world
from tests.integration.test_console import CSRF

pytestmark = pytest.mark.slow


async def _client(tmp_path, create_sessions):
    world = await make_world(tmp_path)
    world.settings.hidden_port_range = (39000, 39999)
    try:
        for name in create_sessions:
            sid = await world.sup.create(world.request(name=name))
            await world.wait_state(sid, "ready")
        app = create_app(world.sup, "table-test-token", 7860, csrf=CSRF)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                    base_url="http://127.0.0.1:7860") as client:
            yield world, client
    finally:
        await world.sup.shutdown()
        for row in await world.store.sessions():
            await world.sup.tunnels.stop(row["tunnel_pid"], row["colab_name"])
        world.fake.kill_all()
        world.store.db.close()


@pytest.fixture
async def one_live(tmp_path):
    async for world, client in _client(tmp_path, ["llm"]):
        yield world, client


@pytest.fixture
async def empty_world(tmp_path):
    async for world, client in _client(tmp_path, []):
        yield world, client


async def table(client, tab, query=""):
    params = {"tab": tab}
    if query:
        params["q"] = query
    response = await client.get("/ui/sessions/table", params=params)
    assert response.status_code == 200
    return response.text


async def test_live_table_with_a_live_session_shows_the_row(one_live):
    world, client = one_live
    body = await table(client, "live")
    assert "llm" in body
    assert "Nothing is billing through colabinet." not in body


async def test_whitespace_only_search_is_not_a_filter(one_live):
    world, client = one_live
    # The store really does hold a ready session, so a blank search must show it rather
    # than fall through to the "nothing is billing" empty state.
    assert any(row["state"] == "ready" for row in await world.store.live_sessions())
    for query in (" ", "    ", "\t", "  \n "):
        body = await table(client, "live", query)
        assert "llm" in body, query
        assert "Nothing is billing through colabinet." not in body, query


async def test_padded_search_terms_still_match(one_live):
    world, client = one_live
    for query in (" llm ", "  LLM  ", " llm\t"):
        body = await table(client, "live", query)
        assert "llm" in body, query
        assert "No sessions match" not in body, query


async def test_filter_miss_on_a_populated_tab_makes_no_billing_claim(one_live):
    world, client = one_live
    body = await table(client, "live", "no-such-session")
    assert "No sessions match" in body
    # The live tab is populated and its session is billing; the copy must not deny that.
    assert "Nothing is billing through colabinet." not in body
    assert "Sessions are still running" not in body


async def test_history_filter_miss_does_not_imply_live_billing(one_live):
    world, client = one_live
    for row in await world.store.sessions():
        await world.sup.stop(row["id"])
    stopped = await table(client, "all")
    assert "stopped" in stopped
    body = await table(client, "history", "no-such-session")
    assert "No sessions match" in body
    assert "Nothing is billing through colabinet." not in body
    assert "Sessions are still running" not in body


@pytest.mark.parametrize("tab,expected", [
    ("live", "No live sessions"),
    ("history", "No stopped sessions yet"),
    ("all", "No sessions yet"),
])
async def test_empty_database_stays_truthful_for_every_tab(empty_world, tab, expected):
    world, client = empty_world
    assert await world.store.sessions() == []
    # A query on a genuinely empty tab must keep reporting that tab's real state rather
    # than the filter-miss copy, which claims rows exist behind the filter.
    body = await table(client, tab, "no-such-session")
    assert expected in body
    assert "No sessions match" not in body
    assert "Sessions are still running" not in body


async def test_empty_database_without_a_query_is_still_truthful(empty_world):
    world, client = empty_world
    for tab in ("live", "history", "all"):
        body = await table(client, tab)
        assert "Sessions are still running" not in body, tab
        assert "No sessions match" not in body, tab