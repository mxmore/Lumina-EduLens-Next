"""
Multi-cloud storage abstraction using the Factory Pattern.

Supports:
- Minio (local/S3-compatible) for development
- Azure Blob Storage for production

Switch providers by setting the STORAGE_MODE environment variable:
  STORAGE_MODE=minio  (default)
  STORAGE_MODE=azure
"""

import io
import os
from abc import ABC, abstractmethod
from typing import BinaryIO, Optional


class StorageProvider(ABC):
    """Abstract base class for object storage providers."""

    @abstractmethod
    async def upload(self, file: BinaryIO, bucket: str, object_name: str) -> str:
        """Upload a file and return its storage URL."""

    @abstractmethod
    async def download(self, bucket: str, object_name: str) -> bytes:
        """Download a file and return its bytes."""

    @abstractmethod
    async def delete(self, bucket: str, object_name: str) -> None:
        """Delete a file from storage."""

    @abstractmethod
    async def get_url(self, bucket: str, object_name: str, expires_seconds: int = 3600) -> str:
        """Return a pre-signed / accessible URL for a stored object."""


class MinioProvider(StorageProvider):
    """Minio storage provider (S3-compatible, for local/dev environments)."""

    def __init__(self) -> None:
        from minio import Minio  # type: ignore

        endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.getenv("MINIO_ROOT_USER", "admin")
        secret_key = os.getenv("MINIO_ROOT_PASSWORD", "")
        secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def _ensure_bucket(self, bucket: str) -> None:
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    async def upload(self, file: BinaryIO, bucket: str, object_name: str) -> str:
        self._ensure_bucket(bucket)
        data = file.read()
        self.client.put_object(
            bucket,
            object_name,
            io.BytesIO(data),
            length=len(data),
            part_size=10 * 1024 * 1024,
        )
        return f"minio://{bucket}/{object_name}"

    async def download(self, bucket: str, object_name: str) -> bytes:
        response = self.client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def delete(self, bucket: str, object_name: str) -> None:
        self.client.remove_object(bucket, object_name)

    async def get_url(self, bucket: str, object_name: str, expires_seconds: int = 3600) -> str:
        from datetime import timedelta

        return self.client.presigned_get_object(
            bucket, object_name, expires=timedelta(seconds=expires_seconds)
        )


class AzureProvider(StorageProvider):
    """Azure Blob Storage provider (for production environments)."""

    def __init__(self) -> None:
        from azure.storage.blob import BlobServiceClient  # type: ignore

        connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        self.client = BlobServiceClient.from_connection_string(connection_string)

    def _get_blob_client(self, bucket: str, object_name: str):
        return self.client.get_blob_client(container=bucket, blob=object_name)

    def _ensure_container(self, bucket: str) -> None:
        container_client = self.client.get_container_client(bucket)
        if not container_client.exists():
            container_client.create_container()

    async def upload(self, file: BinaryIO, bucket: str, object_name: str) -> str:
        self._ensure_container(bucket)
        blob_client = self._get_blob_client(bucket, object_name)
        blob_client.upload_blob(file, overwrite=True)
        account_name = self.client.account_name
        return f"https://{account_name}.blob.core.windows.net/{bucket}/{object_name}"

    async def download(self, bucket: str, object_name: str) -> bytes:
        blob_client = self._get_blob_client(bucket, object_name)
        downloader = blob_client.download_blob()
        return downloader.readall()

    async def delete(self, bucket: str, object_name: str) -> None:
        blob_client = self._get_blob_client(bucket, object_name)
        blob_client.delete_blob()

    async def get_url(self, bucket: str, object_name: str, expires_seconds: int = 3600) -> str:
        from datetime import datetime, timedelta, timezone

        from azure.storage.blob import BlobSasPermissions, generate_blob_sas  # type: ignore

        expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_seconds)
        sas_token = generate_blob_sas(
            account_name=self.client.account_name,
            container_name=bucket,
            blob_name=object_name,
            account_key=self.client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        account_name = self.client.account_name
        return (
            f"https://{account_name}.blob.core.windows.net/{bucket}/{object_name}?{sas_token}"
        )


class StorageFactory:
    """Factory that selects a storage provider based on the STORAGE_MODE env var."""

    _instance: Optional[StorageProvider] = None

    @classmethod
    def get_provider(cls) -> StorageProvider:
        if cls._instance is None:
            mode = os.getenv("STORAGE_MODE", "minio").lower()
            if mode == "azure":
                cls._instance = AzureProvider()
            else:
                cls._instance = MinioProvider()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the cached instance (useful for testing)."""
        cls._instance = None
