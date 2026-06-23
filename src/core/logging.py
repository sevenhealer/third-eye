from __future__ import annotations

import logging
import sys

import structlog
from structlog.types import EventDict


def _add_severity(
    logger: logging.Logger, method: str, event_dict: EventDict
) -> EventDict:
    event_dict["severity"] = event_dict.get("level", method).upper()
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        # logger.exception(...)/log.error(..., exc_info=True) set an
        # exc_info flag that JSONRenderer alone can't serialize - without
        # this, every traceback in the whole app silently vanishes into
        # just `"exc_info": true` (found while debugging the camera
        # supervisor). format_exc_info renders it into an "exception" key.
        structlog.processors.format_exc_info,
        _add_severity,
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Silence noisy libraries
    for noisy in ("uvicorn.access", "aiokafka", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
