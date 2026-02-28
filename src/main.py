"""
FastAPI application – EduLens-Next REST API.

Endpoints:
  POST /students                    Create a student profile
  GET  /students/{id}               Get student profile + knowledge map
  POST /students/{id}/upload        Upload a homework/exam photo
  POST /students/{id}/sessions      Start a study session
  PUT  /students/{id}/sessions/{s}  End a study session
  GET  /students/{id}/errors        List error items
  POST /students/{id}/boosters      Generate booster questions for an error
  POST /students/{id}/plan          Generate / fetch today's study plan
  GET  /students/{id}/alerts        Drain pending alerts
  GET  /students/{id}/reports       List weekly reports
  GET  /health                      Health check
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agents.edu_agent import EduAgent
from src.storage.db import Database, apply_schema
from src.storage.redis_client import RedisClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Application lifespan                                                 #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting EduLens-Next API…")
    await Database.connect()
    RedisClient.connect()
    if os.getenv("AUTO_MIGRATE", "true").lower() == "true":
        await apply_schema()
    yield
    logger.info("Shutting down EduLens-Next API…")
    await Database.disconnect()
    await RedisClient.close()


app = FastAPI(
    title="EduLens-Next API",
    description="AI-powered adaptive learning companion for children",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Pydantic models                                                      #
# ------------------------------------------------------------------ #

class StudentCreate(BaseModel):
    name: str
    birthday: Optional[str] = None
    grade: Optional[str] = None
    personality_tags: Optional[Dict[str, Any]] = None


class SessionStart(BaseModel):
    material_id: Optional[str] = None


class SessionEnd(BaseModel):
    accuracy_rate: Optional[float] = None
    cognitive_load: Optional[float] = None
    status: str = "completed"


class StudyPlanRequest(BaseModel):
    homework_list: List[str]
    energy_level: str = "medium"
    plan_date: Optional[str] = None


class BoosterRequest(BaseModel):
    error_item_id: str
    difficulty: int = 3


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


# ---------- Students ----------

@app.post("/students", status_code=201)
async def create_student(body: StudentCreate) -> Dict[str, Any]:
    birthday = None
    if body.birthday:
        from datetime import date
        birthday = date.fromisoformat(body.birthday)

    student_id = await Database.fetchval(
        """
        INSERT INTO student_profile (name, birthday, grade, personality_tags)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        """,
        body.name,
        birthday,
        body.grade,
        body.personality_tags or {},
    )
    return {"id": str(student_id), "name": body.name}


@app.get("/students/{student_id}")
async def get_student(student_id: str) -> Dict[str, Any]:
    row = await Database.fetchrow(
        "SELECT * FROM student_profile WHERE id = $1", student_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="Student not found")

    # Attach knowledge map
    km_rows = await Database.fetch(
        """
        SELECT kn.subject, kn.topic, skm.mastery_score, skm.status
        FROM student_knowledge_map skm
        JOIN knowledge_node kn ON kn.id = skm.knowledge_id
        WHERE skm.student_id = $1
        ORDER BY kn.subject, skm.status
        """,
        student_id,
    )
    result = dict(row)
    result["id"] = str(result["id"])
    result["knowledge_map"] = [dict(r) for r in km_rows]
    return result


# ---------- Upload ----------

@app.post("/students/{student_id}/upload")
async def upload_material(
    student_id: str,
    file: UploadFile = File(...),
    file_type: str = Form("homework"),
) -> Dict[str, Any]:
    """Upload and analyse a homework/exam photo."""
    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    agent = EduAgent()
    result = await agent.process_new_upload(
        student_id=student_id,
        image_bytes=image_bytes,
        filename=file.filename or "upload.jpg",
        file_type=file_type,
    )
    return result


# ---------- Sessions ----------

@app.post("/students/{student_id}/sessions", status_code=201)
async def start_session(student_id: str, body: SessionStart) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    session_id = await Database.fetchval(
        """
        INSERT INTO study_session (student_id, material_id, start_time, status)
        VALUES ($1, $2, $3, 'in_progress')
        RETURNING id
        """,
        student_id,
        body.material_id,
        now,
    )
    await RedisClient.start_session(student_id, str(session_id))
    return {"session_id": str(session_id), "start_time": now.isoformat()}


@app.put("/students/{student_id}/sessions/{session_id}")
async def end_session(
    student_id: str, session_id: str, body: SessionEnd
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    await Database.execute(
        """
        UPDATE study_session
        SET end_time = $1, accuracy_rate = $2, cognitive_load = $3, status = $4
        WHERE id = $5 AND student_id = $6
        """,
        now,
        body.accuracy_rate,
        body.cognitive_load,
        body.status,
        session_id,
        student_id,
    )
    await RedisClient.end_session(student_id)
    return {"session_id": session_id, "end_time": now.isoformat(), "status": body.status}


# ---------- Errors ----------

@app.get("/students/{student_id}/errors")
async def list_errors(
    student_id: str,
    resolved: Optional[bool] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    if resolved is None:
        rows = await Database.fetch(
            "SELECT * FROM error_item WHERE student_id = $1 ORDER BY created_at DESC LIMIT $2",
            student_id,
            limit,
        )
    else:
        rows = await Database.fetch(
            "SELECT * FROM error_item WHERE student_id = $1 AND resolved = $2 ORDER BY created_at DESC LIMIT $3",
            student_id,
            resolved,
            limit,
        )
    return [
        {**dict(r), "id": str(r["id"]), "student_id": str(r["student_id"])}
        for r in rows
    ]


# ---------- Booster questions ----------

@app.post("/students/{student_id}/boosters")
async def generate_boosters(student_id: str, body: BoosterRequest) -> Dict[str, Any]:
    agent = EduAgent()
    return await agent.generate_boosters(
        student_id=student_id,
        error_item_id=body.error_item_id,
        difficulty=body.difficulty,
    )


# ---------- Study plan ----------

@app.post("/students/{student_id}/plan")
async def generate_plan(student_id: str, body: StudyPlanRequest) -> Dict[str, Any]:
    agent = EduAgent()
    return await agent.generate_daily_plan(
        student_id=student_id,
        homework_list=body.homework_list,
        energy_level=body.energy_level,
        plan_date=body.plan_date,
    )


@app.get("/students/{student_id}/plan")
async def get_plan(student_id: str, plan_date: Optional[str] = None) -> Dict[str, Any]:
    from datetime import date

    target = date.fromisoformat(plan_date) if plan_date else date.today()
    row = await Database.fetchrow(
        "SELECT * FROM daily_study_plan WHERE student_id = $1 AND plan_date = $2",
        student_id,
        target,
    )
    if not row:
        raise HTTPException(status_code=404, detail="No plan found for this date")
    return dict(row)


# ---------- Alerts ----------

@app.get("/students/{student_id}/alerts")
async def get_alerts(student_id: str) -> List[Dict[str, Any]]:
    """Drain and return all pending alerts for a student."""
    return await RedisClient.pop_alerts(student_id)


# ---------- Reports ----------

@app.get("/students/{student_id}/reports")
async def list_reports(student_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    rows = await Database.fetch(
        """
        SELECT * FROM weekly_report
        WHERE student_id = $1
        ORDER BY week_start DESC
        LIMIT $2
        """,
        student_id,
        limit,
    )
    return [
        {**dict(r), "id": str(r["id"]), "student_id": str(r["student_id"])}
        for r in rows
    ]
