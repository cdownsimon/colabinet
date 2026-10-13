import pytest

from colabinet.billing import RECLAIM_S, close_time, session_cu, spend_between
from colabinet.config import Clock, cli_flags, detected_class


def row(created, stopped=None, rate=6.0):
    return {"created_at": created, "stopped_at": stopped, "rate_cu_h": rate}


def test_session_cu_open_and_closed():
    assert session_cu(row(0, None), now=1800) == pytest.approx(3.0)
    assert session_cu(row(0, 600), now=99999) == pytest.approx(1.0)


def test_spend_splits_at_midnight():
    day = (86400.0, 172800.0)
    r = row(86400 - 1800, 86400 + 1800)  # 30 min before and after midnight
    assert spend_between([r], *day, now=10**9) == pytest.approx(3.0)
    assert spend_between([r], 0.0, 86400.0, now=10**9) == pytest.approx(3.0)


def test_close_time_rules():
    assert close_time(now=500, created_at=100, vm_created=False, reason="user", last_exec_ok=None) == 100
    assert close_time(now=500, created_at=100, vm_created=True, reason="user", last_exec_ok=None) == 500
    # reclaimed while the app was down: capped at last successful exec + 25 min
    assert close_time(now=100_000, created_at=0, vm_created=True, reason="lost", last_exec_ok=1000) == 1000 + RECLAIM_S
    assert close_time(now=1200, created_at=0, vm_created=True, reason="reclaimed", last_exec_ok=1000) == 1200


@pytest.mark.parametrize(
    "name,vram,ram,expected",
    [
        (None, None, 12, "CPU"),
        ("NVIDIA L4", 23034, 52, "L4"),
        ("NVIDIA A100-SXM4-40GB", 40960, 87, "A100"),
        ("NVIDIA A100-SXM4-80GB", 81920, 175, "A100-80"),
        ("Tesla T4", 15360, 12.7, "T4"),
        ("Tesla T4", 15360, 51, "T4-HIGHMEM"),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 97887, 185, "G4"),
        ("NVIDIA H100 80GB HBM3", 81559, 200, "UNKNOWN"),
    ],
)
def test_detected_class(name, vram, ram, expected):
    assert detected_class(name, vram, ram) == expected


def test_cli_flags():
    assert cli_flags("CPU") == []
    assert cli_flags("A100-80") == ["--gpu", "A100", "--high-mem"]
    with pytest.raises(ValueError):
        cli_flags("H100")


def test_real_clock_day_bounds_contain_t():
    c = Clock()
    t = c.now()
    start, end = c.day_bounds(t)
    assert start <= t < end and 22 * 3600 <= end - start <= 26 * 3600
