from types import SimpleNamespace

from colabinet.config import Settings
from colabinet.web.app import make_templates
from colabinet.web.views import settings


async def test_rates_show_latest_valid_system_ram_for_detected_instance_class(store, tmp_path):
    # The requested class can differ from the hardware Colab actually allocates.
    observations = [("CPU", 12.5), ("L4", 52.0), ("T4-HIGHMEM", 51.0), ("L4", 54.0),
                    ("L4", None), ("L4", 0.0), ("L4", -1.0), ("L4", float("inf"))]
    for index, (gpu, ram_gb) in enumerate(observations):
        await store.insert_session(
            colab_name=f"recorded-{index}", gpu_requested="T4", gpu_class=gpu, ram_gb=ram_gb,
            proxy_port=3080 + index, hidden_port=23000 + index, vm_port=8090, state="stopped",
            state_since=float(index), created_at=float(index), rate_cu_h=1.54,
            rate_source="measured", idle_min=30,
        )

    async def settings_values():
        return {"eligible_gpus": "T4,L4", "daily_cap_cu": 0, "low_balance_cu": 10,
                "idle_min_default": 30, "notify": False}

    supervisor = SimpleNamespace(store=store, s=Settings(data_dir=tmp_path), cli_version=None,
                                 hf_token=None, settings_values=settings_values)
    context = await settings(supervisor)
    assert context["colab_auth"] == "oauth2"
    assert context["eligible_gpus"] == {"T4", "L4"}
    rates = {row["gpu"]: row for row in context["rates"]}
    assert rates["CPU"]["ram_gb"] == 12.5
    assert rates["L4"]["ram_gb"] == 54.0
    assert rates["T4-HIGHMEM"]["ram_gb"] == 51.0
    assert all(rates[gpu]["ram_source"] == "observed" for gpu in ("CPU", "L4", "T4-HIGHMEM"))
    assert rates["T4"]["ram_gb"] == 13.6 and rates["T4"]["ram_source"] == "reference"
    assert rates["G4"]["ram_gb"] == 189.9 and rates["G4"]["ram_source"] == "reference"

    page = make_templates().env.get_template("pages/settings.html").render(
        st=context, csrf="test-csrf", active="settings",
        account={"state": "idle", "mode": "oauth2", "busy": False, "accounts": []},
        base={"balance": None, "burn_rate": 0, "live_count": 0, "alert_count": 0,
              "command_sessions": [], "daily": {"spend": 0, "cap": 10}},
    )
    assert "System RAM" in page and "54.0 GB" in page and "~13.6 GB" in page
    assert "Last observed" in page and "Reference" in page
    assert "Google accounts" in page and "Sign in with Google" in page and "Eligible GPUs" in page
