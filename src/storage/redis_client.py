"""
Redis client for:
  - Short-term session state (active study sessions)
  - Student dashboard cache
  - Delayed-task / alert queues
  - Agent short-term memory (conversation context)
"""

import json
import os
from typing import Any, Dict, Optional

import redis.asyncio as aioredis  # type: ignore


class RedisClient:
    """Async Redis wrapper with typed helpers for common EduLens patterns."""

    _client: Optional[Any] = None  # aioredis.Redis instance

    @classmethod
    def connect(cls) -> None:
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        cls._client = aioredis.from_url(url, decode_responses=True)

    @classmethod
    def client(cls) -> Any:  # returns aioredis.Redis at runtime
        if cls._client is None:
            raise RuntimeError("Redis is not connected. Call RedisClient.connect() first.")
        return cls._client

    @classmethod
    async def close(cls) -> None:
        if cls._client:
            await cls._client.aclose()
            cls._client = None

    # ------------------------------------------------------------------ #
    # Active study session helpers                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    async def start_session(cls, student_id: str, session_id: str, ttl: int = 7200) -> None:
        """Mark a student as actively studying (expires in `ttl` seconds)."""
        key = f"student:{student_id}:active_session"
        await cls.client().set(key, session_id, ex=ttl)
        await cls.client().hset(
            f"student:{student_id}:session_meta",
            mapping={"session_id": session_id, "status": "in_progress"},
        )

    @classmethod
    async def end_session(cls, student_id: str) -> None:
        """Remove the active session marker for a student."""
        await cls.client().delete(f"student:{student_id}:active_session")
        await cls.client().delete(f"student:{student_id}:session_meta")

    @classmethod
    async def get_active_session(cls, student_id: str) -> Optional[str]:
        """Return the active session id for a student, or None."""
        return await cls.client().get(f"student:{student_id}:active_session")

    # ------------------------------------------------------------------ #
    # Dashboard / cache helpers                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    async def cache_dashboard(cls, student_id: str, data: Dict[str, Any], ttl: int = 300) -> None:
        key = f"dashboard:{student_id}"
        await cls.client().set(key, json.dumps(data, ensure_ascii=False), ex=ttl)

    @classmethod
    async def get_dashboard(cls, student_id: str) -> Optional[Dict[str, Any]]:
        key = f"dashboard:{student_id}"
        raw = await cls.client().get(key)
        return json.loads(raw) if raw else None

    # ------------------------------------------------------------------ #
    # Short-term agent memory (conversation context)                       #
    # ------------------------------------------------------------------ #

    @classmethod
    async def push_agent_memory(cls, student_id: str, message: Dict[str, Any], max_items: int = 20) -> None:
        """Append a message to the agent's rolling context buffer."""
        key = f"agent_memory:{student_id}"
        await cls.client().rpush(key, json.dumps(message, ensure_ascii=False))
        await cls.client().ltrim(key, -max_items, -1)
        await cls.client().expire(key, 86400)  # 24-hour TTL

    @classmethod
    async def get_agent_memory(cls, student_id: str) -> list:
        key = f"agent_memory:{student_id}"
        items = await cls.client().lrange(key, 0, -1)
        return [json.loads(i) for i in items]

    @classmethod
    async def clear_agent_memory(cls, student_id: str) -> None:
        await cls.client().delete(f"agent_memory:{student_id}")

    # ------------------------------------------------------------------ #
    # Alert / notification queue                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    async def push_alert(cls, student_id: str, alert: Dict[str, Any]) -> None:
        """Push an alert notification into the student's alert stream."""
        key = f"alerts:{student_id}"
        await cls.client().rpush(key, json.dumps(alert, ensure_ascii=False))
        await cls.client().expire(key, 86400)

    @classmethod
    async def pop_alerts(cls, student_id: str) -> list:
        """Drain and return all pending alerts for a student."""
        key = f"alerts:{student_id}"
        pipe = cls.client().pipeline()
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        results, _ = await pipe.execute()
        return [json.loads(r) for r in results]
