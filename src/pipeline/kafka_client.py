from __future__ import annotations

import json
from typing import Any

from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError

from src.core.config import get_settings
from src.core.exceptions import KafkaPublishError
from src.core.logging import get_logger

logger = get_logger(__name__)

_producer: AIOKafkaProducer | None = None


async def get_producer() -> AIOKafkaProducer:
    global _producer
    if _producer is None:
        settings = get_settings()
        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode(),
            acks="all",
            compression_type="lz4",
            max_batch_size=131072,
            linger_ms=5,
        )
        await _producer.start()
        logger.info("kafka_producer_started")
    return _producer


async def publish(topic: str, message: dict[str, Any], key: str | None = None) -> None:
    try:
        producer = await get_producer()
        key_bytes = key.encode() if key else None
        await producer.send_and_wait(topic, value=message, key=key_bytes)
    except KafkaConnectionError as e:
        raise KafkaPublishError(f"Kafka publish failed for topic {topic}: {e}") from e


async def stop_producer() -> None:
    global _producer
    if _producer:
        await _producer.stop()
        _producer = None
        logger.info("kafka_producer_stopped")


def make_consumer(
    topics: list[str],
    group_id: str | None = None,
) -> AIOKafkaConsumer:
    settings = get_settings()
    return AIOKafkaConsumer(
        *topics,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=group_id or settings.kafka_consumer_group,
        value_deserializer=lambda v: json.loads(v.decode()),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        auto_commit_interval_ms=1000,
    )
