"""
Celery task definitions.

Each task wraps a CronAgent coroutine and runs it in an async context.
"""

import asyncio

from src.celery_app import celery_app
from src.storage.db import Database
from src.storage.redis_client import RedisClient


def _run_async(coro):
    """Run a coroutine in a new event loop (Celery workers are sync)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _ensure_connections():
    """Ensure DB and Redis are connected for the current async context."""
    if Database._pool is None:
        await Database.connect()
    if RedisClient._client is None:
        RedisClient.connect()


@celery_app.task(name="src.tasks.check_session_speeds")
def check_session_speeds():
    from src.agents.cron_agent import CronAgent

    async def _run():
        await _ensure_connections()
        return await CronAgent().check_session_speeds()

    return _run_async(_run())


@celery_app.task(name="src.tasks.push_tomorrow_plans")
def push_tomorrow_plans():
    from src.agents.cron_agent import CronAgent

    async def _run():
        await _ensure_connections()
        return await CronAgent().push_tomorrow_plans()

    return _run_async(_run())


@celery_app.task(name="src.tasks.generate_weekly_reports")
def generate_weekly_reports():
    from src.agents.cron_agent import CronAgent

    async def _run():
        await _ensure_connections()
        return await CronAgent().generate_weekly_reports()

    return _run_async(_run())
