import asyncio
import logging
import re
import shutil
from dataclasses import dataclass


LOGGER = logging.getLogger(__name__)
TUNNEL_URL_PATTERN = re.compile(r"https://[\w-]+\.trycloudflare\.com")


def get_websocket_url(tunnel_url: str) -> str:
    return f"wss://{tunnel_url.removeprefix('https://')}/ws/analyze"


@dataclass
class QuickTunnel:
    process: asyncio.subprocess.Process
    log_tasks: list[asyncio.Task[None]]


async def _read_tunnel_output(stream: asyncio.StreamReader) -> None:
    while line := await stream.readline():
        message = line.decode(errors="replace").strip()
        match = TUNNEL_URL_PATTERN.search(message)
        if match:
            tunnel_url = match.group(0)
            LOGGER.info("Tunnel URL: %s", tunnel_url)
            LOGGER.info("WebSocket URL: %s", get_websocket_url(tunnel_url))


async def start_quick_tunnel(target_url: str = "http://127.0.0.1:8000") -> QuickTunnel | None:
    executable = shutil.which("cloudflared")
    if not executable:
        LOGGER.warning("cloudflared is not installed; tunnel URL will not be created.")
        return None

    process = await asyncio.create_subprocess_exec(
        executable,
        "tunnel",
        "--url",
        target_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    log_tasks = [
        asyncio.create_task(_read_tunnel_output(stream))
        for stream in (process.stdout, process.stderr)
        if stream is not None
    ]
    return QuickTunnel(process=process, log_tasks=log_tasks)


async def stop_quick_tunnel(tunnel: QuickTunnel | None) -> None:
    if tunnel is None:
        return

    if tunnel.process.returncode is None:
        tunnel.process.terminate()
        try:
            await asyncio.wait_for(tunnel.process.wait(), timeout=5)
        except TimeoutError:
            tunnel.process.kill()
            await tunnel.process.wait()

    for task in tunnel.log_tasks:
        task.cancel()
    await asyncio.gather(*tunnel.log_tasks, return_exceptions=True)
