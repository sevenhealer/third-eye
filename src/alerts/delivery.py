from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)

ALERT_CHANNEL = "thirdeye:alerts"


def _post_webhook(url: str, body: bytes, timeout: float = 3.0) -> None:
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout):
        pass


class AlertDelivery:
    """
    Fans a triggered alert out to (a) the configured webhook URL and
    (b) the redis pub/sub channel the API's WebSocket endpoint relays.

    Delivery is fire-and-forget from the pipeline's perspective: failures
    are logged and retried (webhook: 3 attempts, exponential backoff) but
    NEVER raise into the frame loop — a dead listener must not stall
    detection.
    """

    def __init__(self, webhook_url: str = "") -> None:
        self.webhook_url = webhook_url
        self._tasks: set[asyncio.Task] = set()

    def deliver(self, alert: dict[str, Any]) -> None:
        """Schedule async delivery; returns immediately."""
        task = asyncio.get_running_loop().create_task(self._deliver(alert))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deliver(self, alert: dict[str, Any]) -> None:
        alert = {**alert, "delivered_at": time.time()}
        body = json.dumps(alert, default=str).encode()

        try:
            from src.core.database import get_redis
            redis = await get_redis()
            await redis.publish(ALERT_CHANNEL, body)
        except Exception as exc:
            logger.warning("alert_redis_publish_failed", error=str(exc))

        if self.webhook_url:
            for attempt, delay in enumerate((0.0, 0.5, 1.0), start=1):
                if delay:
                    await asyncio.sleep(delay)
                try:
                    await asyncio.to_thread(_post_webhook, self.webhook_url, body)
                    logger.info("alert_webhook_delivered",
                                url=self.webhook_url, attempt=attempt)
                    return
                except (urllib.error.URLError, OSError) as exc:
                    logger.warning("alert_webhook_failed",
                                   url=self.webhook_url, attempt=attempt,
                                   error=str(exc))
            logger.error("alert_webhook_gave_up", url=self.webhook_url)

    async def drain(self, timeout: float = 5.0) -> None:
        """Let in-flight deliveries finish on shutdown."""
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)
