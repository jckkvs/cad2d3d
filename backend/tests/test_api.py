"""
FastAPI API ルートのユニットテスト.

テスト対象:
- ヘルスチェック
- ファイルアップロード
- エンジン一覧
- 出力フォーマット一覧
"""
from __future__ import annotations

import io
import pytest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.engines.registry import EngineRegistry


@pytest.fixture(autouse=True)
def reset():
    """テスト前にレジストリをリセット."""
    EngineRegistry.reset()
    EngineRegistry._discovered = True  # 自動検出をスキップ
    yield
    EngineRegistry.reset()


@pytest.fixture
def client():
    """FastAPI テストクライアント."""
    return TestClient(app)


class TestHealthCheck:
    """ヘルスチェック API."""

    def test_health(self, client: TestClient) -> None:
        """/api/health が 200 を返す."""
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestUploadAPI:
    """ファイルアップロード API."""

    def test_upload_image(self, client: TestClient, tmp_path: Path) -> None:
        """画像ファイルをアップロードできる."""
        with patch("app.api.routes.upload.settings") as mock_settings:
            mock_settings.upload_dir = tmp_path / "uploads"
            mock_settings.upload_dir.mkdir(parents=True, exist_ok=True)
            mock_settings.max_upload_size_mb = 200
            mock_settings.allowed_image_extensions = [".jpg", ".png"]
            mock_settings.allowed_cad_extensions = [".dxf"]
            mock_settings.allowed_document_extensions = [".pdf"]
            mock_settings.ensure_dirs = lambda: None

            # テスト用の小さいファイル
            file_content = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # 偽 JPEG ヘッダ
            res = client.post(
                "/api/upload/",
                files=[("files", ("test.jpg", io.BytesIO(file_content), "image/jpeg"))],
            )
            assert res.status_code == 200
            data = res.json()
            assert data["success"] is True
            assert len(data["data"]) == 1
            assert data["data"][0]["original_name"] == "test.jpg"

    def test_upload_unsupported_format(self, client: TestClient, tmp_path: Path) -> None:
        """サポート外の形式は 400 エラー."""
        with patch("app.api.routes.upload.settings") as mock_settings:
            mock_settings.upload_dir = tmp_path / "uploads"
            mock_settings.upload_dir.mkdir(parents=True, exist_ok=True)
            mock_settings.max_upload_size_mb = 200
            mock_settings.allowed_image_extensions = [".jpg", ".png"]
            mock_settings.allowed_cad_extensions = [".dxf"]
            mock_settings.allowed_document_extensions = [".pdf"]
            mock_settings.ensure_dirs = lambda: None

            res = client.post(
                "/api/upload/",
                files=[("files", ("test.exe", io.BytesIO(b"malware"), "application/octet-stream"))],
            )
            assert res.status_code == 400


class TestEngineAPI:
    """エンジン API."""

    def test_list_engines_empty(self, client: TestClient) -> None:
        """エンジン未登録時は空リスト."""
        res = client.get("/api/generate/engines")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)

    def test_get_nonexistent_engine(self, client: TestClient) -> None:
        """存在しないエンジンは 404."""
        res = client.get("/api/generate/engines/nonexistent")
        assert res.status_code == 404


class TestExportAPI:
    """エクスポート API."""

    def test_list_formats(self, client: TestClient) -> None:
        """出力フォーマット一覧を返す."""
        res = client.get("/api/export/formats")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        values = [f["value"] for f in data["data"]]
        assert "stl" in values
        assert "obj" in values
        assert "glb" in values

    def test_download_nonexistent_job(self, client: TestClient) -> None:
        """存在しないジョブは 404."""
        res = client.get("/api/export/download/nonexistent-id")
        assert res.status_code == 404
