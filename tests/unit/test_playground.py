import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from jinja2 import Environment, FileSystemLoader

from colabinet.config import PACKAGE_DIR
from colabinet.web.services import send_playground


@pytest.mark.parametrize("engine", ["llamacpp", "exllamav3"])
@pytest.mark.parametrize("thinking", ["none", "low", "medium", "high", "xhigh"])
async def test_playground_thinking_controls_the_shipped_template(engine, thinking):
    captured = {}

    def respond(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": []})

    store = SimpleNamespace(session=AsyncMock(return_value={"id": 1, "state": "ready", "proxy_port": 3080}),
                            current_run=AsyncMock(return_value={"alias": "model", "engine": engine}))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        await send_playground(SimpleNamespace(http=http, store=store),
                              {"session_id": 1, "thinking": thinking, "prompt": "Hello", "stream": False})
    template = Environment(loader=FileSystemLoader(PACKAGE_DIR / "chat_templates")).get_template(
        "qwen38_effort_map.jinja")
    rendered = template.render(messages=captured["messages"], add_generation_prompt=True,
                               **captured.get("chat_template_kwargs", {}))
    if thinking == "none":
        assert rendered.endswith("<think>\n\n</think>\n\n")
        assert "Reasoning effort is set" not in rendered
    else:
        assert rendered.endswith("<think>\n")
        assert captured["chat_template_kwargs"]["reasoning_effort"] == thinking
        if thinking != "medium":
            expected = "low" if thinking == "low" else "xhigh"
            assert f"Reasoning effort is set to {expected}." in rendered
