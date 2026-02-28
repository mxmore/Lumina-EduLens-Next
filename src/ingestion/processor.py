"""
Image ingestion pipeline.

Responsibilities:
1. Pre-process uploaded images (normalise, de-noise).
2. Run OCR to extract text from homework / exam photos.
3. Segment the extracted content into structured items
   (question stem, student answer, mark).
4. Compute a semantic embedding for the extracted content
   to support RAG-based retrieval later.
"""

import base64
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ExtractedItem:
    """A single question / answer pair extracted from an image."""
    question_text: str
    student_answer: Optional[str] = None
    is_marked_correct: Optional[bool] = None   # None = unknown
    page_number: int = 1
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


@dataclass
class IngestionResult:
    """Full result of processing one uploaded image."""
    raw_text: str
    items: List[ExtractedItem] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    language: str = "zh"                       # detected language
    ocr_confidence: float = 0.0                # 0–1 confidence score


class ImageProcessor:
    """
    Processes homework/exam images using Azure AI Vision OCR
    and GPT-4o multimodal analysis.

    If AZURE_VISION_ENDPOINT / AZURE_VISION_KEY are set, the Azure
    Read API is used for high-accuracy OCR.  Otherwise the image is
    sent directly to GPT-4o for combined OCR + analysis.
    """

    def __init__(self) -> None:
        self._azure_endpoint = os.getenv("AZURE_VISION_ENDPOINT")
        self._azure_key = os.getenv("AZURE_VISION_KEY")
        self._openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self._openai_key = os.getenv("AZURE_OPENAI_KEY")
        self._openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    async def process(self, image_bytes: bytes, filename: str = "") -> IngestionResult:
        """Full pipeline: OCR → segment → embed."""
        raw_text, confidence = await self._ocr(image_bytes, filename)
        items = self._segment(raw_text)
        embedding = await self._embed(raw_text)
        return IngestionResult(
            raw_text=raw_text,
            items=items,
            embedding=embedding,
            ocr_confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    # OCR                                                                  #
    # ------------------------------------------------------------------ #

    async def _ocr(self, image_bytes: bytes, filename: str) -> Tuple[str, float]:
        if self._azure_endpoint and self._azure_key:
            return await self._azure_ocr(image_bytes)
        return await self._gpt4o_ocr(image_bytes, filename)

    async def _azure_ocr(self, image_bytes: bytes) -> Tuple[str, float]:
        """Use Azure AI Vision Read API for OCR."""
        from azure.ai.vision.imageanalysis import ImageAnalysisClient  # type: ignore
        from azure.ai.vision.imageanalysis.models import VisualFeatures  # type: ignore
        from azure.core.credentials import AzureKeyCredential  # type: ignore

        client = ImageAnalysisClient(
            endpoint=self._azure_endpoint,
            credential=AzureKeyCredential(self._azure_key),
        )
        result = client.analyze(
            image_data=image_bytes,
            visual_features=[VisualFeatures.READ],
        )
        lines: List[str] = []
        confidence_values: List[float] = []
        if result.read:
            for block in result.read.blocks:
                for line in block.lines:
                    lines.append(line.text)
                    for word in line.words:
                        confidence_values.append(word.confidence)
        raw_text = "\n".join(lines)
        avg_confidence = (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        )
        return raw_text, avg_confidence

    async def _gpt4o_ocr(self, image_bytes: bytes, filename: str) -> Tuple[str, float]:
        """Use GPT-4o vision capabilities as a fallback OCR engine."""
        from openai import AsyncAzureOpenAI  # type: ignore

        b64 = base64.b64encode(image_bytes).decode("utf-8")
        ext = os.path.splitext(filename)[-1].lower().lstrip(".") or "jpeg"
        mime = f"image/{ext}" if ext in ("png", "gif", "webp") else "image/jpeg"

        client = AsyncAzureOpenAI(
            azure_endpoint=self._openai_endpoint,
            api_key=self._openai_key,
            api_version="2024-02-01",
        )
        response = await client.chat.completions.create(
            model=self._openai_deployment,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "请完整提取这张作业/试卷图片中的所有文字内容，"
                                "保持原有的题目编号和结构，逐题列出题目和学生的手写答案。"
                                "输出格式为纯文本，每道题单独一行。"
                            ),
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )
        raw_text = response.choices[0].message.content or ""
        return raw_text, 0.85  # estimated confidence for GPT-4o OCR

    # ------------------------------------------------------------------ #
    # Segmentation                                                         #
    # ------------------------------------------------------------------ #

    def _segment(self, raw_text: str) -> List[ExtractedItem]:
        """
        Heuristic segmentation: split the raw OCR text into individual
        question/answer pairs.

        Looks for numbered question patterns (e.g. "1.", "（1）", "第1题").
        """
        if not raw_text.strip():
            return []

        # Common question-number patterns in Chinese/English homework
        pattern = re.compile(
            r"(?:^|\n)\s*(?:第\s*)?(\d+)[.、。）\)]\s*",
            re.MULTILINE,
        )
        splits = list(pattern.finditer(raw_text))

        if not splits:
            # No numbered questions found; treat entire text as one item
            return [ExtractedItem(question_text=raw_text.strip())]

        items: List[ExtractedItem] = []
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(raw_text)
            block = raw_text[start:end].strip()
            # Attempt to split question from student answer on "答：" or "=" markers
            answer_split = re.split(r"答[：:]\s*|=\s*", block, maxsplit=1)
            question_text = answer_split[0].strip()
            student_answer = answer_split[1].strip() if len(answer_split) > 1 else None
            # Detect correction marks (✓/×/√/×)
            is_correct: Optional[bool] = None
            if re.search(r"[✓√]", block):
                is_correct = True
            elif re.search(r"[✗×✕]", block):
                is_correct = False
            items.append(
                ExtractedItem(
                    question_text=question_text,
                    student_answer=student_answer,
                    is_marked_correct=is_correct,
                )
            )
        return items

    # ------------------------------------------------------------------ #
    # Embedding                                                            #
    # ------------------------------------------------------------------ #

    async def _embed(self, text: str) -> Optional[List[float]]:
        """Generate a 1536-dim embedding vector using the Azure OpenAI embeddings API."""
        if not (self._openai_endpoint and self._openai_key):
            return None
        if not text.strip():
            return None

        from openai import AsyncAzureOpenAI  # type: ignore

        embed_deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
        client = AsyncAzureOpenAI(
            azure_endpoint=self._openai_endpoint,
            api_key=self._openai_key,
            api_version="2024-02-01",
        )
        # Truncate to first 8000 chars to stay within token limits
        response = await client.embeddings.create(
            input=text[:8000],
            model=embed_deployment,
        )
        return response.data[0].embedding
