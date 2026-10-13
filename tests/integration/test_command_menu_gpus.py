"""The command palette may only offer GPUs that the New-session form can actually select.

The palette used to carry its own hardcoded list, so unchecking G4 in Settings removed the
radio from the form while "New session on G4" kept linking to /new?gpu=G4 — a destination
the form silently answered with a different class. Every assertion here goes through the
real settings endpoint and the real page responses, so the old hardcoded menu fails them.
"""

import re

import httpx
import pytest

from colabinet.api import create_app
from tests.integration.conftest import make_world
from tests.integration.test_console import CSRF

pytestmark = pytest.mark.slow

MENU_GPU = re.compile(r'href="/new\?gpu=([A-Za-z0-9-]+)"')
FORM_GPU = re.compile(r'name="gpu" value="([A-Za-z0-9-]+)"')

PAGES = ["/", "/sessions", "/new", "/playground", "/metrics", "/settings"]


@pytest.fixture
async def console(tmp_path):
    world = await make_world(tmp_path)
    app = create_app(world.sup, "menu-test-token", 7860, csrf=CSRF)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://127.0.0.1:7860") as client:
        yield world, client
    await world.sup.shutdown()
    world.fake.kill_all()
    world.store.db.close()


async def get(client, path):
    response = await client.get(path)
    assert response.status_code == 200, path
    return response.text


async def set_eligible(client, gpus):
    """Save Settings the way the checkbox form does: only checked boxes are submitted."""
    response = await client.post("/ui/actions/settings", headers={"X-CSRF-Token": CSRF},
                                 data={"eligible_gpus": list(gpus), "notify": "on"})
    assert response.status_code == 200, response.text
    return response.text


def menu_gpus(body):
    return MENU_GPU.findall(body)


def form_gpus(body):
    return FORM_GPU.findall(body)


async def allowed_gpus(client):
    """The GPU radios the New-session form is currently offering."""
    return form_gpus(await get(client, "/new"))


async def test_default_settings_offer_every_configured_gpu(console):
    world, client = console
    assert "G4" in (await world.sup.settings_values())["eligible_gpus"]
    assert set(menu_gpus(await get(client, "/"))) == set(await allowed_gpus(client))


async def test_unchecking_g4_removes_it_from_the_palette(console):
    """The reported repro: G4 unchecked in Settings, still offered in the palette."""
    world, client = console
    await set_eligible(client, ["A100", "L4", "T4"])
    assert "G4" not in await allowed_gpus(client)
    for page in PAGES:
        gpus = menu_gpus(await get(client, page))
        assert "G4" not in gpus, page
        assert gpus, page


async def test_palette_and_form_agree_after_every_settings_change(console):
    world, client = console
    for eligible in (["A100", "L4", "T4"], ["L4"], ["G4", "T4"], ["A100", "G4"], ["G4", "A100", "L4", "T4"]):
        await set_eligible(client, eligible)
        allowed = await allowed_gpus(client)
        for page in PAGES:
            offered = menu_gpus(await get(client, page))
            assert offered == allowed, (eligible, page)


async def test_family_variants_are_offered_when_the_family_is_enabled(console):
    world, client = console
    await set_eligible(client, ["A100", "T4"])
    assert {"A100", "A100-80", "T4", "T4-HIGHMEM", "CPU"} <= set(await allowed_gpus(client))
    for page in PAGES:
        assert set(menu_gpus(await get(client, page))) >= {"A100", "A100-80", "T4", "T4-HIGHMEM", "CPU"}, page


async def test_disabling_a_family_drops_its_variants_from_the_palette(console):
    world, client = console
    await set_eligible(client, ["L4"])
    assert await allowed_gpus(client) == ["CPU", "L4"]
    for page in PAGES:
        offered = menu_gpus(await get(client, page))
        assert offered == ["CPU", "L4"], page
        assert not {"A100", "A100-80", "T4", "T4-HIGHMEM", "G4"} & set(offered), page


async def test_cpu_only_configuration_still_offers_cpu(console):
    world, client = console
    await set_eligible(client, [])
    assert await allowed_gpus(client) == ["CPU"]
    for page in PAGES:
        assert menu_gpus(await get(client, page)) == ["CPU"], page


async def test_every_offered_command_selects_the_gpu_it_names(console):
    """Each palette entry must land on the form with that GPU already selected."""
    world, client = console
    for eligible in (["A100", "L4", "T4"], ["L4"], [], ["G4", "A100", "L4", "T4"]):
        await set_eligible(client, eligible)
        for gpu in menu_gpus(await get(client, "/")):
            body = await get(client, f"/new?gpu={gpu}")
            selected = re.findall(r'name="gpu" value="([A-Za-z0-9-]+)" checked', body)
            assert selected == [gpu], (eligible, gpu, selected)


async def test_menu_never_offers_a_gpu_the_form_would_replace(console):
    """The fallback on /new?gpu=<disabled> stays, but the menu must not walk into it."""
    world, client = console
    for eligible in (["A100", "L4", "T4"], ["L4"], []):
        await set_eligible(client, eligible)
        allowed = set(await allowed_gpus(client))
        for gpu in menu_gpus(await get(client, "/")):
            assert gpu in allowed, (eligible, gpu)
            body = await get(client, f"/new?gpu={gpu}")
            assert f'name="gpu" value="{gpu}" checked' in body, (eligible, gpu)


async def test_settings_persistence_is_reflected_after_a_reload(console):
    world, client = console
    await set_eligible(client, ["L4"])
    # A fresh request must not see a cached palette from before the save.
    for _ in range(2):
        body = await get(client, "/")
        assert menu_gpus(body) == ["CPU", "L4"]
        assert "G4" not in body.split('id="cmdk-new"')[1].split("</div>")[0]


async def test_saving_settings_returns_the_refreshed_palette_in_place(console):
    """The save response carries the new group, so the open page can swap it without a reload.

    The form posts with hx-swap="none", so a 204 would leave the open document holding the
    pre-save list until something else happened to re-render the page.
    """
    world, client = console
    for eligible, expected in ((["A100", "L4", "T4"], ["CPU", "T4", "T4-HIGHMEM", "L4", "A100", "A100-80"]),
                               (["L4"], ["CPU", "L4"]),
                               ([], ["CPU"]),
                               (["G4", "A100", "L4", "T4"],
                                ["CPU", "T4", "T4-HIGHMEM", "L4", "A100", "A100-80", "G4"])):
        body = await set_eligible(client, eligible)
        assert 'id="cmdk-gpus"' in body, eligible
        assert 'hx-swap-oob="outerHTML"' in body, eligible
        assert menu_gpus(body) == expected, eligible
