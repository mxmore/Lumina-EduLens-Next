"""
Unit tests for EduLens-Next.

These tests run without any live services (no PostgreSQL, Redis, or
Azure OpenAI connections required).  External dependencies are patched
with lightweight mocks.
"""

import io
import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy optional dependencies so tests run without installing them
# ---------------------------------------------------------------------------

def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# minio
minio_mod = _stub_module("minio", Minio=MagicMock)
_stub_module("minio.error", MinioException=Exception)

# azure-storage-blob
_stub_module("azure", storage=MagicMock())
_stub_module("azure.storage", blob=MagicMock())
_stub_module(
    "azure.storage.blob",
    BlobServiceClient=MagicMock(),
    BlobSasPermissions=MagicMock(),
    generate_blob_sas=MagicMock(return_value="sas=token"),
)
_stub_module("azure.core", credentials=MagicMock())
_stub_module("azure.core.credentials", AzureKeyCredential=MagicMock())
_stub_module("azure.ai", vision=MagicMock())
_stub_module("azure.ai.vision", imageanalysis=MagicMock())
_stub_module("azure.ai.vision.imageanalysis", ImageAnalysisClient=MagicMock(), models=MagicMock())
_stub_module("azure.ai.vision.imageanalysis.models", VisualFeatures=MagicMock())

# asyncpg
asyncpg_mod = _stub_module("asyncpg", Pool=MagicMock(), Connection=MagicMock(), Record=dict)
asyncpg_mod.create_pool = AsyncMock()

# redis
redis_mod = _stub_module("redis")
redis_asyncio_mod = _stub_module("redis.asyncio")
redis_asyncio_mod.from_url = MagicMock(return_value=AsyncMock())

# openai
openai_mod = _stub_module("openai")
openai_mod.AsyncAzureOpenAI = MagicMock()

# celery
celery_mod = _stub_module("celery", Celery=MagicMock())
celery_mod.schedules = _stub_module("celery.schedules", crontab=MagicMock())
_stub_module("celery.schedules", crontab=MagicMock())


# ===========================================================================
# Tests: StorageFactory
# ===========================================================================

class TestStorageFactory:

    def setup_method(self):
        from src.storage.storage_service import StorageFactory
        StorageFactory.reset()

    def test_default_provider_is_minio(self):
        os.environ.pop("STORAGE_MODE", None)
        from src.storage.storage_service import MinioProvider, StorageFactory
        provider = StorageFactory.get_provider()
        assert isinstance(provider, MinioProvider)

    def test_azure_provider_selected_when_env_set(self, monkeypatch):
        monkeypatch.setenv("STORAGE_MODE", "azure")
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net")
        from src.storage.storage_service import AzureProvider, StorageFactory
        StorageFactory.reset()
        provider = StorageFactory.get_provider()
        assert isinstance(provider, AzureProvider)

    def test_factory_caches_instance(self):
        from src.storage.storage_service import StorageFactory
        p1 = StorageFactory.get_provider()
        p2 = StorageFactory.get_provider()
        assert p1 is p2

    def test_reset_clears_cache(self):
        from src.storage.storage_service import StorageFactory
        p1 = StorageFactory.get_provider()
        StorageFactory.reset()
        p2 = StorageFactory.get_provider()
        assert p1 is not p2


# ===========================================================================
# Tests: MinioProvider
# ===========================================================================

class TestMinioProvider:

    def _make_provider(self):
        from src.storage.storage_service import MinioProvider
        provider = MinioProvider.__new__(MinioProvider)
        provider.client = MagicMock()
        provider.client.bucket_exists = MagicMock(return_value=True)
        provider.client.put_object = MagicMock()
        provider.client.get_object = MagicMock()
        provider.client.remove_object = MagicMock()
        provider.client.presigned_get_object = MagicMock(return_value="http://minio/presigned")
        return provider

    @pytest.mark.asyncio
    async def test_upload_returns_minio_url(self):
        provider = self._make_provider()
        mock_response = MagicMock()
        mock_response.read.return_value = b"data"
        provider.client.get_object.return_value = mock_response

        url = await provider.upload(io.BytesIO(b"hello"), "bucket", "file.jpg")
        assert url == "minio://bucket/file.jpg"
        provider.client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_creates_bucket_if_missing(self):
        provider = self._make_provider()
        provider.client.bucket_exists.return_value = False
        provider.client.make_bucket = MagicMock()

        await provider.upload(io.BytesIO(b"x"), "new-bucket", "obj.jpg")
        provider.client.make_bucket.assert_called_once_with("new-bucket")

    @pytest.mark.asyncio
    async def test_delete_calls_remove_object(self):
        provider = self._make_provider()
        await provider.delete("bucket", "file.jpg")
        provider.client.remove_object.assert_called_once_with("bucket", "file.jpg")

    @pytest.mark.asyncio
    async def test_get_url_returns_presigned(self):
        provider = self._make_provider()
        url = await provider.get_url("bucket", "file.jpg")
        assert url == "http://minio/presigned"


# ===========================================================================
# Tests: ImageProcessor segmentation (no external calls needed)
# ===========================================================================

class TestImageProcessorSegmentation:

    def _make_processor(self):
        from src.ingestion.processor import ImageProcessor
        p = ImageProcessor.__new__(ImageProcessor)
        p._azure_endpoint = None
        p._azure_key = None
        p._openai_endpoint = None
        p._openai_key = None
        p._openai_deployment = "gpt-4o"
        return p

    def test_segment_empty_text(self):
        p = self._make_processor()
        assert p._segment("") == []

    def test_segment_numbered_questions(self):
        p = self._make_processor()
        text = "1. 三角形内角和等于多少？答：180度\n2. 平行四边形面积如何计算？答：底×高"
        items = p._segment(text)
        assert len(items) == 2
        assert "三角形" in items[0].question_text
        assert items[0].student_answer == "180度"
        assert "平行四边形" in items[1].question_text

    def test_segment_detects_correct_mark(self):
        p = self._make_processor()
        text = "1. 2+2=？答：4 ✓"
        items = p._segment(text)
        assert items[0].is_marked_correct is True

    def test_segment_detects_incorrect_mark(self):
        p = self._make_processor()
        text = "1. 3×3=？答：6 ✗"
        items = p._segment(text)
        assert items[0].is_marked_correct is False

    def test_segment_no_numbers_returns_single_item(self):
        p = self._make_processor()
        text = "这是一段没有题号的自由文本。"
        items = p._segment(text)
        assert len(items) == 1
        assert items[0].question_text == text


# ===========================================================================
# Tests: Prompt builder functions
# ===========================================================================

class TestPromptBuilders:

    def test_build_error_attribution_messages_structure(self):
        from src.brain.prompts import build_error_attribution_messages
        msgs = build_error_attribution_messages(
            question_text="1+1=?",
            student_answer="3",
            correct_answer="2",
            history_context=[{"question_text": "2+2=?", "error_category": "calculation"}],
            personality_tags={"patience": 0.5},
        )
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "1+1=?" in msgs[1]["content"]
        assert "3" in msgs[1]["content"]

    def test_build_study_plan_messages_structure(self):
        from src.brain.prompts import build_study_plan_messages
        msgs = build_study_plan_messages(
            homework_list=["数学练习册 P3"],
            knowledge_map={"三角形内角和": "red"},
            energy_level="high",
            avg_speed=25.0,
            personality_tags={},
            plan_date="2026-02-28",
        )
        assert msgs[0]["role"] == "system"
        assert "2026-02-28" in msgs[1]["content"]
        assert "三角形内角和" in msgs[1]["content"]

    def test_build_booster_messages_structure(self):
        from src.brain.prompts import build_booster_messages
        msgs = build_booster_messages(
            original_question="两数之和为10",
            error_analysis={"error_category": "logic_gap"},
            knowledge_points=["方程求解"],
            difficulty=3,
        )
        assert msgs[0]["role"] == "system"
        assert "方程求解" in msgs[1]["content"]
        assert "3/5" in msgs[1]["content"]


# ===========================================================================
# Tests: RedisClient helpers (mocked)
# ===========================================================================

class TestRedisClientHelpers:

    @pytest.mark.asyncio
    async def test_start_and_get_active_session(self):
        from src.storage.redis_client import RedisClient

        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value="session-abc")
        RedisClient._client = mock_redis

        await RedisClient.start_session("student-1", "session-abc")
        result = await RedisClient.get_active_session("student-1")
        assert result == "session-abc"

    @pytest.mark.asyncio
    async def test_push_and_pop_alerts(self):
        from src.storage.redis_client import RedisClient

        stored = []

        mock_redis = AsyncMock()
        mock_redis.rpush = AsyncMock(side_effect=lambda k, v: stored.append(v))

        async def fake_pipeline():
            pipe = MagicMock()
            pipe.lrange = MagicMock()
            pipe.delete = MagicMock()
            pipe.execute = AsyncMock(return_value=[stored[:], None])
            return pipe

        mock_redis.pipeline = MagicMock(return_value=MagicMock(
            lrange=MagicMock(),
            delete=MagicMock(),
            execute=AsyncMock(return_value=[
                [json.dumps({"type": "speed_alert", "message": "rest"})],
                None,
            ]),
        ))
        RedisClient._client = mock_redis

        await RedisClient.push_alert("student-1", {"type": "speed_alert", "message": "rest"})
        alerts = await RedisClient.pop_alerts("student-1")
        assert len(alerts) == 1
        assert alerts[0]["type"] == "speed_alert"

    @pytest.mark.asyncio
    async def test_push_agent_memory(self):
        from src.storage.redis_client import RedisClient

        mock_redis = AsyncMock()
        RedisClient._client = mock_redis

        await RedisClient.push_agent_memory("s1", {"role": "user", "content": "hello"})
        mock_redis.rpush.assert_called_once()


# ===========================================================================
# Tests: FastAPI endpoints (using TestClient, no live services)
# ===========================================================================

class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        """Create a FastAPI test client with mocked DB and Redis."""
        from fastapi.testclient import TestClient

        # Patch lifespan dependencies
        with patch("src.main.Database.connect", new_callable=AsyncMock), \
             patch("src.main.Database.disconnect", new_callable=AsyncMock), \
             patch("src.main.apply_schema", new_callable=AsyncMock), \
             patch("src.main.RedisClient.connect"), \
             patch("src.main.RedisClient.close", new_callable=AsyncMock):
            from src.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_create_student(self, client):
        with patch("src.main.Database.fetchval", new_callable=AsyncMock) as mock_fv:
            import uuid
            mock_fv.return_value = str(uuid.uuid4())
            resp = client.post("/students", json={"name": "小明", "grade": "Grade 3"})
            assert resp.status_code == 201
            data = resp.json()
            assert data["name"] == "小明"
            assert "id" in data

    def test_get_student_not_found(self, client):
        with patch("src.main.Database.fetchrow", new_callable=AsyncMock) as mock_fr:
            mock_fr.return_value = None
            resp = client.get("/students/nonexistent-id")
            assert resp.status_code == 404

    def test_list_errors(self, client):
        with patch("src.main.Database.fetch", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []
            resp = client.get("/students/some-id/errors")
            assert resp.status_code == 200
            assert resp.json() == []

    def test_get_alerts(self, client):
        with patch("src.main.RedisClient.pop_alerts", new_callable=AsyncMock) as mock_alerts:
            mock_alerts.return_value = [{"type": "speed_alert", "message": "休息一下"}]
            resp = client.get("/students/some-id/alerts")
            assert resp.status_code == 200
            assert resp.json()[0]["type"] == "speed_alert"
