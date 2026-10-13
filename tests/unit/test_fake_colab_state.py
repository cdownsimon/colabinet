import fcntl
import threading
from concurrent.futures import ThreadPoolExecutor

from tests.fakes import FakeColab


def test_fake_fault_edits_share_the_cli_transaction_lock(tmp_path):
    fake = FakeColab(tmp_path)
    started = threading.Event()

    def edit_fault():
        started.set()
        fake.fault(stop_noop=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with fake.state_path.with_suffix(".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = fake.read()
            state["balance"] = 123.0
            future = executor.submit(edit_fault)
            assert started.wait(1)
            try:
                future.result(timeout=0.1)
            except TimeoutError:
                pass
            fake.write(state)
        future.result(timeout=1)
    state = fake.read()
    assert state["faults"]["stop_noop"] is True
    assert state["balance"] == 123.0
