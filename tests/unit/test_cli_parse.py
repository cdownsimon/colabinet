from colabinet.cli_parse import Runtime, make_listing, parse_sessions, parse_status, parse_usage

SESSIONS = """\
[colab] A new version of google-colab-cli is available: 0.7.5 (you have 0.7.4).
[colab] Pruned 1 stale local session(s).
[llm] gpu-l4-s-abc123 | Hardware: L4 | Shape: Standard | Variant: GPU
[?] gpu-a100-s-zzz | Hardware: A100 | Shape: High-RAM | Variant: GPU
[cpu1] m-s-cpu9 | Hardware: CPU | Shape: Standard | Variant: DEFAULT | Status: IDLE
"""
USAGE = "Current balance: 195.88 compute units\nUsage rate: 1.54/hr\nActive assignments: 3\n"
EMPTY = "[colab] No active sessions found on server.\n"


def test_parse_sessions_skips_noise_and_reads_unnamed_rows():
    rows, said_empty = parse_sessions(SESSIONS)
    assert not said_empty
    assert rows[0] == Runtime("llm", "gpu-l4-s-abc123", "L4", "Standard", "GPU")
    assert rows[1].name is None and rows[1].shape == "High-RAM"
    assert rows[2].hardware == "CPU" and rows[2].variant == "DEFAULT"


def test_parse_usage():
    u = parse_usage(USAGE)
    assert (u.balance, u.rate_per_hour, u.active_assignments) == (195.88, 1.54, 3)
    assert parse_usage("[colab] Warning: failed to fetch compute-unit info") is None


def test_listing_trust_rules():
    assert make_listing(0, SESSIONS, 0, USAGE, 1.0).trusted
    # empty list + usage says 0 -> trusted
    zero = USAGE.replace("Active assignments: 3", "Active assignments: 0")
    assert make_listing(0, EMPTY, 0, zero, 1.0).trusted
    # empty list but usage says 1 (auth failure hidden by an empty store) -> NOT trusted
    one = USAGE.replace("Active assignments: 3", "Active assignments: 1")
    assert not make_listing(0, EMPTY, 0, one, 1.0).trusted
    # usage failed -> not trusted, even with rows
    assert not make_listing(0, SESSIONS, 1, "", 1.0).trusted
    # sessions failed -> not trusted
    assert not make_listing(1, "", 0, zero, 1.0).trusted


def test_listing_lookup():
    listing = make_listing(0, SESSIONS, 0, USAGE, 1.0)
    assert listing.named("llm").endpoint == "gpu-l4-s-abc123"
    assert listing.named("nope") is None
    assert [r.endpoint for r in listing.unnamed] == ["gpu-a100-s-zzz"]


def test_parse_status_takes_last_marker_line():
    out = 'noise\nCOLABINET_STATUS {"a": 1}\nCOLABINET_STATUS {"a": 2}\n'
    assert parse_status(out) == {"a": 2}
    assert parse_status("RuntimeError: Connection was lost.") is None
    assert parse_status("COLABINET_STATUS {broken") is None
