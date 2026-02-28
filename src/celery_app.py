"""
Celery application for background and scheduled tasks.

Workers consume tasks from the Redis broker.
Celery Beat schedules periodic jobs.
"""

import os

from celery import Celery  # type: ignore
from celery.schedules import crontab  # type: ignore

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "edulens",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    beat_schedule={
        # Check session speeds every 10 minutes
        "check-session-speeds": {
            "task": "src.tasks.check_session_speeds",
            "schedule": crontab(minute="*/10"),
        },
        # Push tomorrow's plans at 21:00 every evening
        "push-tomorrow-plans": {
            "task": "src.tasks.push_tomorrow_plans",
            "schedule": crontab(hour=21, minute=0),
        },
        # Generate weekly reports every Monday at 09:00
        "generate-weekly-reports": {
            "task": "src.tasks.generate_weekly_reports",
            "schedule": crontab(hour=9, minute=0, day_of_week="monday"),
        },
    },
)
