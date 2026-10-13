"""Parsers against real `colab` output, with account identifiers replaced (tests/fixtures/cli/)."""

from pathlib import Path

import pytest

from colabinet.cli_parse import make_listing, parse_sessions, parse_usage

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "cli"
pytestmark = pytest.mark.skipif(not (FIX / "sessions_one.txt").exists(), reason="captured from a real CLI run")


def test_real_sessions_output_has_our_cpu_runtime():
    rows, _ = parse_sessions((FIX / "sessions_one.txt").read_text())
    ours = [r for r in rows if r.name == "cbcheck"]
    assert ours and ours[0].hardware == "CPU" and ours[0].endpoint


def test_real_usage_output_parses():
    u = parse_usage((FIX / "usage_one.txt").read_text())
    assert u is not None and u.active_assignments >= 1 and u.rate_per_hour > 0


def test_real_empty_listing_is_trusted_only_with_zero_assignments():
    empty, usage0 = (FIX / "sessions_empty.txt").read_text(), (FIX / "usage_empty.txt").read_text()
    assert make_listing(0, empty, 0, usage0, 0.0).trusted == (parse_usage(usage0).active_assignments == 0)


def test_real_new_output_says_ready():
    assert "Session READY." in (FIX / "new_ok.txt").read_text()
