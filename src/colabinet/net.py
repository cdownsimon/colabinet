"""Small network helpers: port checks and a uvicorn server that leaves signals to us."""

from __future__ import annotations

import asyncio
import contextlib
import socket

import uvicorn


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def first_free(start: int, stop: int, taken: set[int] = frozenset()) -> int:
    for port in range(start, stop + 1):
        if port not in taken and port_free(port):
            return port
    raise OSError(f"no free port in {start}-{stop}")


class QuietServer(uvicorn.Server):
    """uvicorn installs SIGINT/SIGTERM handlers per server; with several servers in one loop they overwrite each
    other. colabinet handles signals itself, so this server never touches them."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def start_server(app, port: int, host: str = "127.0.0.1", *,
                       lifespan: bool = False) -> tuple[QuietServer, asyncio.Task]:
    config = uvicorn.Config(app, host=host, port=port, lifespan="on" if lifespan else "off",
                            log_level="warning", access_log=False,
                            timeout_graceful_shutdown=5)
    server = QuietServer(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            with contextlib.suppress(BaseException):
                task.result()  # uvicorn exits with SystemExit when it cannot bind
            raise OSError(f"could not listen on {host}:{port}")
        await asyncio.sleep(0.01)
    return server, task


async def stop_server(server: QuietServer, task: asyncio.Task) -> None:
    """Stop a server. Never swallows the caller's own cancellation (that once kept a cancelled loop alive)."""
    server.should_exit = True
    try:
        await asyncio.wait_for(asyncio.shield(task), 10)
    except TimeoutError:
        server.force_exit = True
        task.cancel()
    except asyncio.CancelledError:
        if asyncio.current_task().cancelling():
            raise  # we are being cancelled: propagate
        # otherwise the server task itself was cancelled, which is also a stop
    except (Exception, SystemExit):  # the server task failed while stopping: it is stopped either way
        pass
