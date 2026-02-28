"""
RAG (Retrieval-Augmented Generation) engine.

Uses pgvector cosine similarity search to surface relevant context
(past errors, textbook excerpts, notes) before calling the LLM.

Flow:
  1. Embed the current query / question text.
  2. Run cosine similarity search against stored embeddings.
  3. Return top-k snippets as additional context for LLM prompts.
"""

import os
from typing import Any, Dict, List, Optional

from src.storage.db import Database


class RAGEngine:
    """
    Hybrid retrieval engine:
      - Error history lookup (find similar past wrong answers)
      - Textbook / study material semantic search
    """

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k
        self._embed_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._embed_key = os.getenv("AZURE_OPENAI_KEY")
        self._embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")

    # ------------------------------------------------------------------ #
    # Embedding helper                                                     #
    # ------------------------------------------------------------------ #

    async def _embed(self, text: str) -> Optional[List[float]]:
        """Generate an embedding vector for the given text."""
        if not (self._embed_endpoint and self._embed_key):
            return None
        if not text.strip():
            return None

        from openai import AsyncAzureOpenAI  # type: ignore

        client = AsyncAzureOpenAI(
            azure_endpoint=self._embed_endpoint,
            api_key=self._embed_key,
            api_version="2024-02-01",
        )
        response = await client.embeddings.create(
            input=text[:8000],
            model=self._embed_deployment,
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------ #
    # Retrieval methods                                                    #
    # ------------------------------------------------------------------ #

    async def retrieve_similar_errors(
        self,
        student_id: str,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the most semantically similar past errors for a student,
        to provide historical context to the LLM.
        """
        k = top_k or self._top_k
        embedding = await self._embed(query_text)
        if embedding is None:
            # Fallback: return most recent errors without vector search
            rows = await Database.fetch(
                """
                SELECT id, question_text, student_answer, error_category, root_cause, created_at
                FROM error_item
                WHERE student_id = $1 AND resolved = FALSE
                ORDER BY created_at DESC
                LIMIT $2
                """,
                student_id,
                k,
            )
            return [dict(r) for r in rows]

        # pgvector cosine similarity search
        vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
        rows = await Database.fetch(
            f"""
            SELECT id, question_text, student_answer, error_category, root_cause, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM error_item
            WHERE student_id = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            vector_str,
            student_id,
            k,
        )
        return [dict(r) for r in rows]

    async def retrieve_relevant_materials(
        self,
        student_id: str,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find the most semantically relevant study materials (textbooks,
        notes) to enrich LLM context.
        """
        k = top_k or self._top_k
        embedding = await self._embed(query_text)
        if embedding is None:
            rows = await Database.fetch(
                """
                SELECT id, title, file_type, ocr_text, storage_path, upload_date
                FROM study_material
                WHERE student_id = $1
                ORDER BY upload_date DESC
                LIMIT $2
                """,
                student_id,
                k,
            )
            return [dict(r) for r in rows]

        vector_str = "[" + ",".join(str(v) for v in embedding) + "]"
        rows = await Database.fetch(
            f"""
            SELECT id, title, file_type, ocr_text, storage_path, upload_date,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM study_material
            WHERE student_id = $2
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            vector_str,
            student_id,
            k,
        )
        return [dict(r) for r in rows]

    async def build_context_for_error(
        self,
        student_id: str,
        question_text: str,
    ) -> Dict[str, Any]:
        """
        Build a complete RAG context dict for error attribution:
          - similar past errors (history_context)
          - relevant material snippets (material_context)
        """
        similar_errors, relevant_materials = await _gather(
            self.retrieve_similar_errors(student_id, question_text),
            self.retrieve_relevant_materials(student_id, question_text),
        )
        return {
            "history_context": similar_errors,
            "material_context": [
                {"title": m.get("title"), "snippet": (m.get("ocr_text") or "")[:400]}
                for m in relevant_materials
            ],
        }


async def _gather(*coros):
    """Run coroutines concurrently using asyncio.gather."""
    import asyncio

    return await asyncio.gather(*coros)
