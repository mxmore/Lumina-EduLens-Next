"""
LLM client wrapper for Azure OpenAI.

Provides typed helpers for the EduLens-specific inference tasks:
  - Error attribution analysis
  - Study plan generation
  - Booster question generation
  - Weekly report synthesis
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import AsyncAzureOpenAI  # type: ignore

from src.brain.prompts import (
    build_booster_messages,
    build_error_attribution_messages,
    build_study_plan_messages,
)


class LLMClient:
    """Async wrapper around Azure OpenAI Chat Completions."""

    def __init__(self) -> None:
        self._client = AsyncAzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
        self._deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self._temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self._max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    async def _chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request and parse the JSON response."""
        response = await self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens if max_tokens is not None else self._max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: wrap raw string in a dict so callers always get a dict
            return {"raw_response": content}

    # ------------------------------------------------------------------ #
    # Public inference methods                                             #
    # ------------------------------------------------------------------ #

    async def analyze_error(
        self,
        question_text: str,
        student_answer: Optional[str],
        correct_answer: Optional[str],
        history_context: List[Dict[str, Any]],
        personality_tags: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform deep error attribution for a single wrong answer."""
        messages = build_error_attribution_messages(
            question_text=question_text,
            student_answer=student_answer,
            correct_answer=correct_answer,
            history_context=history_context,
            personality_tags=personality_tags,
        )
        return await self._chat(messages)

    async def generate_study_plan(
        self,
        homework_list: List[str],
        knowledge_map: Dict[str, str],
        energy_level: str,
        avg_speed: float,
        personality_tags: Dict[str, Any],
        plan_date: str,
    ) -> Dict[str, Any]:
        """Generate a dynamic daily study plan."""
        messages = build_study_plan_messages(
            homework_list=homework_list,
            knowledge_map=knowledge_map,
            energy_level=energy_level,
            avg_speed=avg_speed,
            personality_tags=personality_tags,
            plan_date=plan_date,
        )
        return await self._chat(messages)

    async def generate_booster_questions(
        self,
        original_question: str,
        error_analysis: Dict[str, Any],
        knowledge_points: List[str],
        difficulty: int = 3,
    ) -> Dict[str, Any]:
        """Generate variant practice questions targeting a specific weakness."""
        messages = build_booster_messages(
            original_question=original_question,
            error_analysis=error_analysis,
            knowledge_points=knowledge_points,
            difficulty=difficulty,
        )
        return await self._chat(messages)

    async def freeform(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.5,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """Generic chat completion for ad-hoc agent tasks."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._chat(messages, temperature=temperature, max_tokens=max_tokens)
