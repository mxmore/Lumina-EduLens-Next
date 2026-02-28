"""
EduAgent – core agent controller.

Orchestrates:
  1. Upload processing (OCR → analyse → store)
  2. Error attribution with RAG context
  3. Study plan generation
  4. Booster question creation
  5. Dashboard updates
"""

import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from src.brain.llm_client import LLMClient
from src.brain.rag import RAGEngine
from src.ingestion.processor import ImageProcessor
from src.storage.db import Database
from src.storage.redis_client import RedisClient
from src.storage.storage_service import StorageFactory


class EduAgent:
    """
    High-level agent that coordinates all sub-systems for a single
    student's learning interactions.
    """

    def __init__(self) -> None:
        self._storage = StorageFactory.get_provider()
        self._processor = ImageProcessor()
        self._llm = LLMClient()
        self._rag = RAGEngine()

    # ------------------------------------------------------------------ #
    # Upload & analyse homework photo                                      #
    # ------------------------------------------------------------------ #

    async def process_new_upload(
        self,
        student_id: str,
        image_bytes: bytes,
        filename: str,
        file_type: str = "homework",
    ) -> Dict[str, Any]:
        """
        Full pipeline for a newly uploaded homework/exam photo:
          1. Store original image in object storage.
          2. OCR + segment the image.
          3. Embed and persist the material record.
          4. Analyse each wrong answer with RAG-augmented LLM.
          5. Store errors and update the student's knowledge map.
          6. Invalidate dashboard cache.
        Returns a summary of the analysis.
        """
        import io

        # 1. Persist to object storage
        object_name = f"{student_id}/{date.today().isoformat()}/{uuid.uuid4().hex}_{filename}"
        url = await self._storage.upload(io.BytesIO(image_bytes), "homework-photos", object_name)

        # 2. OCR + segment
        ingestion_result = await self._processor.process(image_bytes, filename)

        # 3. Store material metadata in PostgreSQL
        material_id = await Database.fetchval(
            """
            INSERT INTO study_material
                (student_id, title, file_type, storage_path, provider, ocr_text, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
            RETURNING id
            """,
            student_id,
            filename,
            file_type,
            object_name,
            os.getenv("STORAGE_MODE", "minio").lower(),
            ingestion_result.raw_text,
            (
                "[" + ",".join(str(v) for v in ingestion_result.embedding) + "]"
                if ingestion_result.embedding
                else None
            ),
        )

        # 4. Fetch student profile for personality context
        profile = await self._get_student_profile(student_id)
        personality_tags = profile.get("personality_tags") or {} if profile else {}

        # 5. Analyse each extracted item
        analyses: List[Dict[str, Any]] = []
        for item in ingestion_result.items:
            if item.is_marked_correct is False or (
                item.student_answer and item.is_marked_correct is None
            ):
                rag_ctx = await self._rag.build_context_for_error(
                    student_id, item.question_text
                )
                analysis = await self._llm.analyze_error(
                    question_text=item.question_text,
                    student_answer=item.student_answer,
                    correct_answer=None,
                    history_context=rag_ctx["history_context"],
                    personality_tags=personality_tags,
                )
                # 6. Persist error item
                await self._persist_error(
                    student_id=student_id,
                    question_text=item.question_text,
                    student_answer=item.student_answer,
                    analysis=analysis,
                    embedding=ingestion_result.embedding,
                )
                analyses.append(
                    {"question": item.question_text[:100], "analysis": analysis}
                )

        # 7. Invalidate Redis dashboard cache
        await RedisClient.cache_dashboard(student_id, {}, ttl=1)

        return {
            "material_id": str(material_id),
            "storage_url": url,
            "ocr_confidence": ingestion_result.ocr_confidence,
            "total_items": len(ingestion_result.items),
            "errors_found": len(analyses),
            "analyses": analyses,
        }

    # ------------------------------------------------------------------ #
    # Study plan generation                                                #
    # ------------------------------------------------------------------ #

    async def generate_daily_plan(
        self,
        student_id: str,
        homework_list: List[str],
        energy_level: str = "medium",
        plan_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate and persist a daily study plan for the student."""
        today = plan_date or date.today().isoformat()

        # Fetch knowledge map statuses
        knowledge_map = await self._get_knowledge_map(student_id)

        # Compute historical average speed
        avg_speed = await self._avg_session_speed(student_id)

        # Fetch personality tags
        profile = await self._get_student_profile(student_id)
        personality_tags = profile.get("personality_tags") or {} if profile else {}

        plan = await self._llm.generate_study_plan(
            homework_list=homework_list,
            knowledge_map=knowledge_map,
            energy_level=energy_level,
            avg_speed=avg_speed,
            personality_tags=personality_tags,
            plan_date=today,
        )

        # Persist the plan
        await Database.execute(
            """
            INSERT INTO daily_study_plan (student_id, plan_date, plan_content)
            VALUES ($1, $2, $3)
            ON CONFLICT (student_id, plan_date) DO UPDATE SET plan_content = EXCLUDED.plan_content
            """,
            student_id,
            date.fromisoformat(today),
            plan,
        )

        return plan

    # ------------------------------------------------------------------ #
    # Booster question generation                                          #
    # ------------------------------------------------------------------ #

    async def generate_boosters(
        self,
        student_id: str,
        error_item_id: str,
        difficulty: int = 3,
    ) -> Dict[str, Any]:
        """Generate variant practice questions for a specific error item."""
        row = await Database.fetchrow(
            "SELECT * FROM error_item WHERE id = $1 AND student_id = $2",
            error_item_id,
            student_id,
        )
        if not row:
            return {"error": "Error item not found"}

        knowledge_points: List[str] = []
        if row["knowledge_id"]:
            kn_row = await Database.fetchrow(
                "SELECT topic FROM knowledge_node WHERE id = $1", row["knowledge_id"]
            )
            if kn_row:
                knowledge_points = [kn_row["topic"]]

        result = await self._llm.generate_booster_questions(
            original_question=row["question_text"],
            error_analysis=row["remediation"] or {},
            knowledge_points=knowledge_points,
            difficulty=difficulty,
        )

        # Persist generated items
        for q in result.get("booster_questions", []):
            await Database.execute(
                """
                INSERT INTO practice_item
                    (student_id, knowledge_id, source_error_id, question_text,
                     expected_answer, difficulty)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                student_id,
                row["knowledge_id"],
                error_item_id,
                q.get("question_text", ""),
                q.get("expected_answer"),
                q.get("difficulty", difficulty),
            )

        return result

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    async def _get_student_profile(self, student_id: str) -> Optional[Dict[str, Any]]:
        row = await Database.fetchrow(
            "SELECT * FROM student_profile WHERE id = $1", student_id
        )
        return dict(row) if row else None

    async def _get_knowledge_map(self, student_id: str) -> Dict[str, str]:
        rows = await Database.fetch(
            """
            SELECT kn.topic, skm.status
            FROM student_knowledge_map skm
            JOIN knowledge_node kn ON kn.id = skm.knowledge_id
            WHERE skm.student_id = $1
            """,
            student_id,
        )
        return {r["topic"]: r["status"] for r in rows}

    async def _avg_session_speed(self, student_id: str) -> float:
        val = await Database.fetchval(
            """
            SELECT AVG(duration_seconds)
            FROM study_session
            WHERE student_id = $1 AND status = 'completed'
            """,
            student_id,
        )
        return float(val) / 60.0 if val else 30.0  # default 30 min

    async def _persist_error(
        self,
        student_id: str,
        question_text: str,
        student_answer: Optional[str],
        analysis: Dict[str, Any],
        embedding: Optional[List[float]],
    ) -> None:
        diagnosis = analysis.get("diagnosis", {})
        remediation = analysis.get("remediation_strategy", {})
        await Database.execute(
            """
            INSERT INTO error_item
                (student_id, question_text, student_answer, error_category,
                 root_cause, remediation, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
            """,
            student_id,
            question_text,
            student_answer,
            diagnosis.get("error_category"),
            diagnosis.get("root_cause"),
            remediation,
            (
                "[" + ",".join(str(v) for v in embedding) + "]"
                if embedding
                else None
            ),
        )
