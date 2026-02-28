"""
CronAgent – scheduled background monitoring agent.

Responsibilities:
  1. Every 10 minutes: detect students who are taking much longer than
     usual and push a rest/help alert via Redis.
  2. Every evening (21:00): scan all students and generate tomorrow's
     daily study plan.
  3. Every Monday: generate weekly assessment reports.

Designed to be triggered by Celery Beat or any cron scheduler.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

from src.brain.llm_client import LLMClient
from src.brain.prompts import EDUCATOR_SYSTEM_PROMPT, WEEKLY_REPORT_PROMPT
from src.storage.db import Database
from src.storage.redis_client import RedisClient

logger = logging.getLogger(__name__)


class CronAgent:
    """Runs background monitoring and scheduled AI tasks."""

    # Factor above which a session is considered "struggling"
    SPEED_ALERT_THRESHOLD = 1.5

    def __init__(self) -> None:
        self._llm = LLMClient()

    # ------------------------------------------------------------------ #
    # 1. Session speed monitoring (every 10 min)                          #
    # ------------------------------------------------------------------ #

    async def check_session_speeds(self) -> int:
        """
        Scan Redis for students with active sessions.
        If a session is running for > SPEED_ALERT_THRESHOLD × their
        historical average, push a rest/help notification.

        Returns the number of alerts generated.
        """
        # Retrieve all active session keys
        client = RedisClient.client()
        keys = await client.keys("student:*:active_session")
        alert_count = 0

        for key in keys:
            student_id = key.split(":")[1]
            session_id = await client.get(key)
            if not session_id:
                continue

            # Get session start time from PostgreSQL
            session_row = await Database.fetchrow(
                "SELECT start_time FROM study_session WHERE id = $1", session_id
            )
            if not session_row:
                continue

            elapsed_minutes = (
                datetime.now(timezone.utc) - session_row["start_time"]
            ).total_seconds() / 60

            # Get historical average (minutes per session)
            avg_minutes = await Database.fetchval(
                """
                SELECT COALESCE(AVG(duration_seconds) / 60.0, 30)
                FROM study_session
                WHERE student_id = $1 AND status = 'completed'
                """,
                student_id,
            )
            avg_minutes = float(avg_minutes)

            if elapsed_minutes > avg_minutes * self.SPEED_ALERT_THRESHOLD:
                await self._push_speed_alert(student_id, elapsed_minutes, avg_minutes)
                alert_count += 1

        logger.info("CronAgent.check_session_speeds: %d alerts generated", alert_count)
        return alert_count

    async def _push_speed_alert(
        self, student_id: str, elapsed: float, average: float
    ) -> None:
        """Generate a contextual LLM hint and push it to the alert queue."""
        result = await self._llm.freeform(
            system_prompt=EDUCATOR_SYSTEM_PROMPT,
            user_message=(
                f"学生已经连续学习了 {elapsed:.0f} 分钟，"
                f"历史平均学习时长为 {average:.0f} 分钟。"
                "请生成一条简短的中文提示，鼓励孩子稍作休息或检查是否需要帮助。"
                "输出 JSON: {\"message\": \"...\", \"type\": \"rest_reminder\"}"
            ),
            temperature=0.7,
            max_tokens=200,
        )
        alert = {
            "type": "speed_alert",
            "message": result.get("message", "你已经学习了很久了，休息一下吧！"),
            "elapsed_minutes": round(elapsed),
            "average_minutes": round(average),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await RedisClient.push_alert(student_id, alert)

    # ------------------------------------------------------------------ #
    # 2. Evening plan push (21:00 daily)                                  #
    # ------------------------------------------------------------------ #

    async def push_tomorrow_plans(self) -> int:
        """
        For every student with a session today, pre-generate tomorrow's
        study plan based on today's performance.
        Returns the number of plans generated.
        """
        today = date.today()
        rows = await Database.fetch(
            """
            SELECT DISTINCT student_id
            FROM study_session
            WHERE DATE(start_time) = $1
            """,
            today,
        )
        count = 0
        for row in rows:
            student_id = str(row["student_id"])
            try:
                await self._generate_plan_for(student_id, today + timedelta(days=1))
                count += 1
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to generate plan for %s: %s", student_id, exc)

        logger.info("CronAgent.push_tomorrow_plans: %d plans generated", count)
        return count

    async def _generate_plan_for(self, student_id: str, plan_date: date) -> None:
        from src.agents.edu_agent import EduAgent

        agent = EduAgent()
        # Get today's incomplete errors as tomorrow's focus areas
        errors = await Database.fetch(
            """
            SELECT DISTINCT kn.topic
            FROM error_item ei
            LEFT JOIN knowledge_node kn ON kn.id = ei.knowledge_id
            WHERE ei.student_id = $1 AND ei.resolved = FALSE
            LIMIT 5
            """,
            student_id,
        )
        homework_list = [r["topic"] for r in errors if r["topic"]] or ["复习今日内容"]
        await agent.generate_daily_plan(
            student_id=student_id,
            homework_list=homework_list,
            plan_date=plan_date.isoformat(),
        )

    # ------------------------------------------------------------------ #
    # 3. Weekly report generation (Monday)                                #
    # ------------------------------------------------------------------ #

    async def generate_weekly_reports(self) -> int:
        """
        Generate weekly assessment reports for all active students.
        Runs on Monday to summarise the previous week.
        Returns the number of reports generated.
        """
        today = date.today()
        week_start = today - timedelta(days=today.weekday() + 7)  # last Monday

        rows = await Database.fetch(
            """
            SELECT DISTINCT student_id
            FROM study_session
            WHERE start_time >= $1 AND start_time < $2
            """,
            week_start,
            week_start + timedelta(days=7),
        )
        count = 0
        for row in rows:
            student_id = str(row["student_id"])
            try:
                await self._generate_weekly_report(student_id, week_start)
                count += 1
            except Exception as exc:  # pragma: no cover
                logger.error("Failed to generate report for %s: %s", student_id, exc)

        logger.info("CronAgent.generate_weekly_reports: %d reports", count)
        return count

    async def _generate_weekly_report(self, student_id: str, week_start: date) -> None:
        week_end = week_start + timedelta(days=7)

        # Aggregate stats
        completion_stats = await Database.fetchrow(
            """
            SELECT
                COUNT(*) AS total_sessions,
                ROUND(AVG(accuracy_rate)::numeric, 2) AS avg_accuracy,
                SUM(duration_seconds) AS total_seconds
            FROM study_session
            WHERE student_id = $1 AND start_time >= $2 AND start_time < $3
            AND status = 'completed'
            """,
            student_id,
            week_start,
            week_end,
        )

        error_summary = await Database.fetch(
            """
            SELECT error_category, COUNT(*) AS cnt
            FROM error_item
            WHERE student_id = $1
              AND created_at >= $2 AND created_at < $3
            GROUP BY error_category
            """,
            student_id,
            week_start,
            week_end,
        )

        user_content = (
            f"**周开始**: {week_start}\n\n"
            f"**完成情况**: {dict(completion_stats) if completion_stats else {}}\n\n"
            f"**错题分类汇总**: {[dict(r) for r in error_summary]}"
        )

        report = await self._llm.freeform(
            system_prompt=EDUCATOR_SYSTEM_PROMPT + "\n\n" + WEEKLY_REPORT_PROMPT,
            user_message=user_content,
            temperature=0.4,
            max_tokens=2000,
        )

        await Database.execute(
            """
            INSERT INTO weekly_report (student_id, week_start, radar_data, summary, recommendations)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (student_id, week_start)
            DO UPDATE SET radar_data = EXCLUDED.radar_data,
                          summary = EXCLUDED.summary,
                          recommendations = EXCLUDED.recommendations
            """,
            student_id,
            week_start,
            report.get("weekly_report", {}).get("radar_data", {}),
            report.get("weekly_report", {}).get("highlights", []),
            report.get("weekly_report", {}).get("next_week_focus", []),
        )
