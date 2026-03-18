"""
カバレッジ向上テスト.

低カバレッジモジュールの追加テスト:
- app.api.routes.export (32%→)
- app.api.routes.settings (47%→)
- app.core.config (87%→)
- app.engines.* (エッジケース)
- app.models.schemas (追加検証)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.schemas import OutputFormat

pytestmark = pytest.mark.usefixtures("ensure_engines")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ===== Core Config テスト =====


class TestCoreConfig:
    """app.core.config のテスト."""

    def test_settings_defaults(self) -> None:
        """デフォルト設定値."""
        from app.core.config import settings
        assert settings.app_name == "CAD3D Generator"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.max_upload_size_mb == 200

    def test_settings_paths(self) -> None:
        """パス設定."""
        from app.core.config import settings
        assert settings.data_dir.name == "data"
        assert settings.weights_dir == settings.data_dir / "weights"
        assert settings.temp_dir == settings.data_dir / "temp"

    def test_allowed_extensions(self) -> None:
        """許可拡張子."""
        from app.core.config import settings
        assert ".png" in settings.allowed_image_extensions
        assert ".jpg" in settings.allowed_image_extensions
        assert ".dxf" in settings.allowed_cad_extensions
        assert ".pdf" in settings.allowed_document_extensions

    def test_network_defaults(self) -> None:
        """ネットワーク設定のデフォルト."""
        from app.core.config import settings
        # デフォルトではプロキシなし
        assert settings.host == "127.0.0.1"
        assert settings.port == 8000

    def test_ensure_dirs(self, tmp_path: Path) -> None:
        """ディレクトリ作成."""
        from app.core.config import Settings
        s = Settings(
            data_dir=tmp_path / "test_data",
            weights_dir=tmp_path / "test_data" / "weights",
            projects_dir=tmp_path / "test_data" / "projects",
            temp_dir=tmp_path / "test_data" / "temp",
            upload_dir=tmp_path / "test_data" / "uploads",
        )
        s.ensure_dirs()
        assert (tmp_path / "test_data").exists()
        assert (tmp_path / "test_data" / "weights").exists()
        assert (tmp_path / "test_data" / "temp").exists()

    def test_cors_origins(self) -> None:
        """CORS設定."""
        from app.core.config import settings
        assert isinstance(settings.cors_origins, list)
        assert len(settings.cors_origins) > 0


# ===== Core Exceptions テスト =====


class TestCoreExceptions:
    """app.core.exceptions のテスト."""

    def test_engine_not_found_error(self) -> None:
        from app.core.exceptions import EngineNotFoundError
        err = EngineNotFoundError("test_engine")
        assert "test_engine" in str(err)

    def test_preprocessing_error(self) -> None:
        from app.core.exceptions import PreprocessingError
        err = PreprocessingError("bad image")
        assert "bad image" in str(err)

    def test_weight_download_error(self) -> None:
        from app.core.exceptions import WeightDownloadError
        err = WeightDownloadError("download failed", detail="timeout")
        assert "download failed" in str(err)

    def test_cad3d_error_base(self) -> None:
        from app.core.exceptions import CAD3DError
        err = CAD3DError("base error")
        assert "base error" in str(err)


# ===== Export API テスト =====


class TestExportAPI:
    """app.api.routes.export のテスト."""

    def test_formats_list(self, client: TestClient) -> None:
        """出力フォーマット一覧."""
        resp = client.get("/api/export/formats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        formats = data["data"]
        assert isinstance(formats, list)
        assert len(formats) > 0
        # OBJ/STL/GLBが含まれる
        values = {f["value"] for f in formats}
        assert "obj" in values
        assert "stl" in values

    def test_download_nonexistent(self, client: TestClient) -> None:
        """存在しないジョブのダウンロード."""
        resp = client.get("/api/export/download/nonexistent-job-xyz")
        assert resp.status_code == 404

    def test_open_external_nonexistent(self, client: TestClient) -> None:
        """存在しないジョブの外部起動."""
        resp = client.post("/api/export/open-external/nonexistent-job-xyz")
        assert resp.status_code == 404

    def test_reload_nonexistent(self, client: TestClient) -> None:
        """存在しないジョブの再読み込み."""
        resp = client.post("/api/export/reload/nonexistent-job-xyz")
        assert resp.status_code == 404


# ===== Settings API テスト =====


class TestSettingsAPI:
    """app.api.routes.settings のテスト."""

    def test_get_settings(self, client: TestClient) -> None:
        """現在の設定取得."""
        resp = client.get("/api/settings/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        settings_data = data["data"]
        assert "proxy" in settings_data
        assert "huggingface" in settings_data

    def test_update_proxy(self, client: TestClient) -> None:
        """プロキシ設定更新."""
        resp = client.put("/api/settings/proxy", json={
            "http_proxy": "http://proxy.example.com:8080",
            "https_proxy": "https://proxy.example.com:8443",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        # 元に戻す
        client.put("/api/settings/proxy", json={
            "http_proxy": None,
            "https_proxy": None,
        })

    def test_update_hf_settings(self, client: TestClient) -> None:
        """HuggingFace設定更新."""
        resp = client.put("/api/settings/huggingface", json={
            "token": "hf_test_token_12345",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_update_hf_masked_token(self, client: TestClient) -> None:
        """マスクされたトークンは更新しない."""
        resp = client.put("/api/settings/huggingface", json={
            "token": "***",
        })
        assert resp.status_code == 200


# ===== Output Format テスト =====


class TestOutputFormat:
    """OutputFormat スキーマのテスト."""

    def test_all_formats_exist(self) -> None:
        """全フォーマットが存在."""
        formats = list(OutputFormat)
        assert len(formats) >= 6
        values = {f.value for f in formats}
        assert "obj" in values
        assert "stl" in values
        assert "glb" in values

    def test_format_values_lowercase(self) -> None:
        """フォーマット値は小文字."""
        for f in OutputFormat:
            assert f.value == f.value.lower()


# ===== Schemas 追加テスト =====


class TestSchemas:
    """app.models.schemas のテスト."""

    def test_api_response_success(self) -> None:
        """APIResponse成功."""
        from app.models.schemas import APIResponse
        resp = APIResponse(success=True, message="OK")
        assert resp.success is True
        assert resp.message == "OK"

    def test_api_response_failure(self) -> None:
        """APIResponse失敗."""
        from app.models.schemas import APIResponse
        resp = APIResponse(success=False, message="error")
        assert resp.success is False
        assert resp.message == "error"

    def test_engine_capabilities_defaults(self) -> None:
        """EngineCapabilities デフォルト値."""
        from app.models.schemas import EngineCapabilities
        caps = EngineCapabilities(
            supports_single_image=True,
            outputs_mesh=True,
            supported_output_formats=[OutputFormat.OBJ],
        )
        assert caps.supports_multi_image is False
        assert caps.outputs_cad is False
        assert caps.requires_gpu is False

    def test_weight_file_info(self) -> None:
        """WeightFileInfo."""
        from app.models.schemas import WeightFileInfo
        w = WeightFileInfo(
            name="model.pth",
            url="https://example.com/model.pth",
            relative_path="engine/model.pth",
            size_bytes=1024 * 1024,
        )
        assert w.name == "model.pth"
        assert w.requires_auth is False  # default

    def test_engine_info(self) -> None:
        """EngineInfo."""
        from app.models.schemas import EngineInfo, EngineCapabilities, EngineStatus
        info = EngineInfo(
            name="test",
            display_name="Test Engine",
            description="テスト",
            version="1.0",
            capabilities=EngineCapabilities(
                supports_single_image=True,
                outputs_mesh=True,
                supported_output_formats=[OutputFormat.OBJ],
            ),
            status=EngineStatus.READY,
            required_weights=[],
        )
        assert info.name == "test"
        assert info.status == EngineStatus.READY

    def test_proxy_settings(self) -> None:
        """ProxySettings."""
        from app.models.schemas import ProxySettings
        ps = ProxySettings()
        assert ps.http_proxy is None
        assert ps.https_proxy is None

    def test_app_settings(self) -> None:
        """AppSettings."""
        from app.models.schemas import AppSettings
        s = AppSettings()
        assert s.proxy is not None
        assert s.huggingface is not None


# ===== エンジンエッジケース =====


class TestEngineEdgeCases:
    """エンジンのエッジケーステスト."""

    def test_registry_list_returns_engine_info(self) -> None:
        """一覧がEngineInfoリスト."""
        from app.engines.registry import EngineRegistry
        engines = EngineRegistry.list_available()
        assert all(hasattr(e, "name") for e in engines)
        assert all(hasattr(e, "status") for e in engines)

    def test_all_engines_have_description(self) -> None:
        """全エンジンに説明文がある."""
        from app.engines.registry import EngineRegistry
        for info in EngineRegistry.list_available():
            assert len(info.description) > 10

    def test_all_engines_have_version(self) -> None:
        """全エンジンにバージョンがある."""
        from app.engines.registry import EngineRegistry
        for info in EngineRegistry.list_available():
            assert len(info.version) > 0

    def test_engines_display_names_unique(self) -> None:
        """表示名が重複しない."""
        from app.engines.registry import EngineRegistry
        names = [e.display_name for e in EngineRegistry.list_available()]
        assert len(names) == len(set(names))

    def test_engine_names_lowercase(self) -> None:
        """エンジン名は小文字."""
        from app.engines.registry import EngineRegistry
        for info in EngineRegistry.list_available():
            assert info.name == info.name.lower()

    def test_at_least_one_gpu_free_engine(self) -> None:
        """GPU不要なエンジンが1つ以上ある."""
        from app.engines.registry import EngineRegistry
        engines = EngineRegistry.list_available()
        gpu_free = [e for e in engines if not e.capabilities.requires_gpu]
        assert len(gpu_free) >= 1

    def test_at_least_one_cad_output_engine(self) -> None:
        """CAD出力対応エンジンが1つ以上ある (SECAD-Net)."""
        from app.engines.registry import EngineRegistry
        engines = EngineRegistry.list_available()
        cad_engines = [e for e in engines if e.capabilities.outputs_cad]
        assert len(cad_engines) >= 1

    def test_all_engines_have_output_formats(self) -> None:
        """全エンジンに出力フォーマットが定義されている."""
        from app.engines.registry import EngineRegistry
        for info in EngineRegistry.list_available():
            assert len(info.capabilities.supported_output_formats) > 0
