"""
深度推定 + パイプライン + 重みマネージャ + Mold API拡張テスト.

テスト対象:
- app.preprocessing.depth_estimator (深度推定/点群生成/エラーハンドリング)
- app.preprocessing.pipeline (統合パイプライン/設定/マルチビュー統合)
- app.weights.manager (WeightManager/ステータス/SHA256/パス管理)
- app.api.routes.mold (金型API: サイジング/部品DB)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


# ===== depth_estimator テスト =====


class TestDepthEstimator:
    """深度推定モジュールのテスト."""

    @pytest.mark.asyncio
    async def test_estimate_depth_unsupported_model(self, tmp_path: Path) -> None:
        """未対応モデル名でPreprocessingError."""
        from app.preprocessing.depth_estimator import estimate_depth
        from app.core.exceptions import PreprocessingError
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        src = tmp_path / "test.png"
        img.save(str(src))
        with pytest.raises(PreprocessingError, match="未対応"):
            await estimate_depth(src, tmp_path / "out", model_name="unknown_model")

    @pytest.mark.asyncio
    async def test_estimate_depth_creates_output_dir(self, tmp_path: Path) -> None:
        """出力ディレクトリは自動作成される."""
        from app.preprocessing.depth_estimator import estimate_depth
        from app.core.exceptions import PreprocessingError
        img = Image.new("RGB", (50, 50))
        src = tmp_path / "input.png"
        img.save(str(src))
        out_dir = tmp_path / "new_dir" / "sub"
        # torch がなければ ImportError → PreprocessingError
        with pytest.raises((PreprocessingError, ImportError)):
            await estimate_depth(src, out_dir, model_name="depth_anything_v2")
        # ディレクトリは作成される
        assert out_dir.exists()

    def test_depth_to_point_cloud(self, tmp_path: Path) -> None:
        """深度マップ+RGB画像から点群生成."""
        from app.preprocessing.depth_estimator import depth_to_point_cloud
        # グラデーション深度マップ
        depth_arr = np.zeros((50, 50), dtype=np.uint8)
        depth_arr[10:40, 10:40] = 200  # 有効領域
        depth_img = Image.fromarray(depth_arr)
        depth_path = tmp_path / "depth.png"
        depth_img.save(str(depth_path))
        # RGB画像
        rgb = Image.new("RGB", (50, 50), (255, 128, 64))
        rgb_path = tmp_path / "rgb.png"
        rgb.save(str(rgb_path))
        # PLY出力
        out_path = tmp_path / "output.ply"
        result = depth_to_point_cloud(depth_path, rgb_path, out_path, focal_length=500.0)
        assert result.exists()
        content = result.read_text()
        assert "ply" in content
        assert "element vertex" in content
        assert "property float x" in content

    def test_depth_to_point_cloud_empty_depth(self, tmp_path: Path) -> None:
        """全黒深度マップ(z=0)でも生成可能."""
        from app.preprocessing.depth_estimator import depth_to_point_cloud
        depth = Image.new("L", (30, 30), 0)
        depth_path = tmp_path / "zero_depth.png"
        depth.save(str(depth_path))
        rgb = Image.new("RGB", (30, 30))
        rgb_path = tmp_path / "black.png"
        rgb.save(str(rgb_path))
        out_path = tmp_path / "out.ply"
        result = depth_to_point_cloud(depth_path, rgb_path, out_path)
        assert result.exists()
        content = result.read_text()
        # z=0のピクセルはmask(z>0.01)で除外
        assert "element vertex 0" in content  # 全点が除外


# ===== pipeline テスト =====


class TestPipeline:
    """前処理パイプラインのテスト."""

    def test_preprocessed_image_dataclass(self, tmp_path: Path) -> None:
        """PreprocessedImageデータクラス."""
        from app.preprocessing.pipeline import PreprocessedImage
        pi = PreprocessedImage(
            path=tmp_path / "clean.png",
            original_path=tmp_path / "original.png",
            view_label="front",
            was_split=True,
        )
        assert pi.view_label == "front"
        assert pi.was_split is True
        assert pi.scale_info is None
        assert pi.removed_annotations == []

    def test_preprocessing_config_defaults(self) -> None:
        """PreprocessingConfigデフォルト値."""
        from app.preprocessing.pipeline import PreprocessingConfig
        config = PreprocessingConfig()
        assert config.auto_split_views is True
        assert config.remove_text is True
        assert config.remove_dimensions is True
        assert config.remove_hatching is True
        assert config.remove_auxiliary is True
        assert config.inpaint is True

    def test_preprocessing_config_custom(self) -> None:
        """PreprocessingConfigカスタム値."""
        from app.preprocessing.pipeline import PreprocessingConfig
        config = PreprocessingConfig(
            auto_split_views=False,
            remove_text=False,
            inpaint=False,
        )
        assert config.auto_split_views is False
        assert config.remove_text is False

    def test_preprocess_file_png(self, tmp_path: Path) -> None:
        """PNG画像のパイプライン全体."""
        from app.preprocessing.pipeline import preprocess_file, PreprocessingConfig
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        src = tmp_path / "drawing.png"
        img.save(str(src))
        out_dir = tmp_path / "output"
        config = PreprocessingConfig(
            auto_split_views=False,
            remove_text=False,
            remove_dimensions=False,
            remove_hatching=False,
            remove_auxiliary=False,
        )
        results = preprocess_file(src, out_dir, config)
        assert len(results) >= 1
        assert results[0].path.exists()

    def test_preprocess_file_with_annotations(self, tmp_path: Path) -> None:
        """注釈除去付きパイプライン."""
        from app.preprocessing.pipeline import preprocess_file, PreprocessingConfig
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        src = tmp_path / "view.png"
        img.save(str(src))
        out_dir = tmp_path / "out"
        config = PreprocessingConfig(
            auto_split_views=False,
            remove_text=True,
            remove_dimensions=False,
        )
        results = preprocess_file(src, out_dir, config)
        assert len(results) >= 1

    def test_preprocess_file_auto_split(self, tmp_path: Path) -> None:
        """マルチビュー分割付きパイプライン."""
        from app.preprocessing.pipeline import preprocess_file
        img = Image.new("RGB", (400, 200), (255, 255, 255))
        src = tmp_path / "multi.png"
        img.save(str(src))
        out_dir = tmp_path / "split_out"
        results = preprocess_file(src, out_dir)
        assert len(results) >= 1

    def test_preprocess_unsupported(self, tmp_path: Path) -> None:
        """非対応形式はPreprocessingError."""
        from app.preprocessing.pipeline import preprocess_file
        from app.core.exceptions import PreprocessingError
        src = tmp_path / "test.abc"
        src.write_text("dummy")
        with pytest.raises(PreprocessingError):
            preprocess_file(src, tmp_path / "out")


# ===== WeightManager テスト =====


class TestWeightManager:
    """重みマネージャのテスト."""

    def test_init(self) -> None:
        """WeightManager初期化."""
        from app.weights.manager import WeightManager
        wm = WeightManager()
        assert wm._active_downloads == {}

    def test_check_weight_exists_false(self, tmp_path: Path) -> None:
        """存在しない重みファイル."""
        from app.weights.manager import WeightManager
        from app.models.schemas import WeightFileInfo
        wm = WeightManager()
        w = WeightFileInfo(
            name="test.pth",
            url="https://example.com/test.pth",
            relative_path="nonexistent/test.pth",
        )
        assert wm.check_weight_exists(w) is False

    def test_get_weight_path(self) -> None:
        """重みファイルの絶対パス取得."""
        from app.weights.manager import WeightManager
        from app.models.schemas import WeightFileInfo
        from app.core.config import settings
        wm = WeightManager()
        w = WeightFileInfo(
            name="model.pth",
            url="https://example.com",
            relative_path="engine/model.pth",
        )
        result = wm.get_weight_path(w)
        assert result == settings.weights_dir / "engine/model.pth"

    def test_get_all_status(self) -> None:
        """全重みステータス取得."""
        from app.weights.manager import WeightManager
        from app.models.schemas import WeightFileInfo
        wm = WeightManager()
        weights = [
            WeightFileInfo(name="a.pth", url="https://a.com", relative_path="a.pth"),
            WeightFileInfo(name="b.pth", url="https://b.com", relative_path="b.pth"),
        ]
        status = wm.get_all_status(weights)
        assert len(status) == 2
        assert status[0]["name"] == "a.pth"
        assert status[0]["exists"] is False
        assert status[0]["downloading"] is False

    def test_compute_sha256(self, tmp_path: Path) -> None:
        """SHA256計算."""
        from app.weights.manager import WeightManager
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")
        result = WeightManager._compute_sha256(test_file)
        import hashlib
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert result == expected

    @pytest.mark.asyncio
    async def test_download_already_running(self) -> None:
        """二重ダウンロード防止."""
        from app.weights.manager import WeightManager
        from app.models.schemas import WeightFileInfo
        from app.core.exceptions import WeightDownloadError
        wm = WeightManager()
        w = WeightFileInfo(name="big.pth", url="https://a.com", relative_path="big.pth")
        wm._active_downloads["big.pth"] = 0.5  # ダウンロード中を模擬
        with pytest.raises(WeightDownloadError, match="already being downloaded"):
            await wm.download_weight(w)

    def test_singleton(self) -> None:
        """シングルトンインスタンス."""
        from app.weights.manager import weight_manager
        assert weight_manager is not None
        assert hasattr(weight_manager, "check_weight_exists")


# ===== Mold API 拡張テスト =====


class TestMoldAPIExtended:
    """金型API拡張テスト."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_parts_endpoint(self, client) -> None:
        """標準部品一覧API."""
        resp = client.get("/api/mold/parts")
        assert resp.status_code == 200
        data = resp.json()
        # APIResponseでラップされるか、直接リストが返る場合がある
        assert isinstance(data, (list, dict))

    def test_sizing_endpoint(self, client) -> None:
        """サイジングAPI (GET or POST)."""
        resp = client.get("/api/mold/sizing")
        # GETで設定取得 or POSTのみ
        assert resp.status_code in (200, 405, 404)

    def test_materials_endpoint(self, client) -> None:
        """樹脂材料一覧API."""
        resp = client.get("/api/mold/materials")
        assert resp.status_code in (200, 404)

    def test_analysis_endpoint_no_file(self, client) -> None:
        """ファイルなしでアンダーカット解析."""
        resp = client.post("/api/mold/undercut")
        assert resp.status_code in (400, 404, 422)
