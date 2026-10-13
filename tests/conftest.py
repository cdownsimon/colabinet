import pytest

from colabinet.db import Database
from colabinet.store import Store


class FakeClock:
    def __init__(self, t: float = 1_800_000_000.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    def day_bounds(self, t: float) -> tuple[float, float]:
        start = t - (t % 86400)
        return start, start + 86400

    def day_key(self, t: float) -> str:
        return str(int(t // 86400))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def store(tmp_path):
    db = Database(tmp_path / "t.db")
    db.init()
    yield Store(db)
    db.close()
