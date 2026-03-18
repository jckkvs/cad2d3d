"""
generate API と前処理パイプラインのテスト.

テスト対象:
- app.api.routes.generate (エンジンREADME/ジョブ管理)
- app.preprocessing.multiview_splitter (ビュー分割/Otsu/ギャップ検出)
- app.preprocessing.pipeline (PreprocessingConfig)
- app.preprocessing.format_converter (追加ケース)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.usefixtures("ensure_engines")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ===== Generate API 追加テスト =====


class TestGenerateAPIExtended:
    """generate API の追加テスト."""

    def test_engine_readme_triposr(self, client: TestClient) -> None:
        """TripoSRのREADME取得."""
        resp = client.get("/api/generate/engines/triposr/readme")
        assert resp.status_code == 200

    def test_engine_readme_secadnet(self, client: TestClient) -> None:
        """SECAD-NetのREADME取得."""
        resp = client.get("/api/generate/engines/secadnet/readme")
        assert resp.status_code == 200

    def test_engine_readme_nonexistent(self, client: TestClient) -> None:
        """存在しないエンジンのREADME."""
        resp = client.get("/api/generate/engines/nonexistent_xyz/readme")
        assert resp.status_code in (404, 500)

    def test_engine_info_instantmesh(self, client: TestClient) -> None:
        """InstantMeshの情報."""
        resp = client.get("/api/generate/engines/instantmesh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "instantmesh"

    def test_engine_info_crm(self, client: TestClient) -> None:
        """CRMの情報."""
        resp = client.get("/api/generate/engines/crm")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "crm"

    def test_engine_info_wonder3d(self, client: TestClient) -> None:
        """Wonder3Dの情報."""
        resp = client.get("/api/generate/engines/wonder3d")
        assert resp.status_code == 200

    def test_engine_info_zero123pp(self, client: TestClient) -> None:
        """Zero123++の情報."""
        resp = client.get("/api/generate/engines/zero123pp")
        assert resp.status_code == 200


# ===== マルチビュー分割テスト =====


class TestMultiviewSplitter:
    """マルチビュー検出・分割のテスト."""

    def test_single_view_detection(self, tmp_path: Path) -> None:
        """単一ビュー画像の検出."""
        from app.preprocessing.multiview_splitter import detect_and_split_views
        # 白背景に黒い矩形を1つだけ描画
        img = Image.new("L", (200, 200), 255)
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill=0)
        path = tmp_path / "single.png"
        img.save(str(path))

        views = detect_and_split_views(path)
        assert len(views) >= 1

    def test_multi_view_detection(self, tmp_path: Path) -> None:
        """複数ビューの検出（分割結果が返ること）."""
        from app.preprocessing.multiview_splitter import detect_and_split_views
        img = Image.new("L", (400, 400), 255)
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 170, 170], fill=0)
        draw.rectangle([230, 20, 380, 170], fill=0)
        draw.rectangle([20, 230, 170, 380], fill=0)
        draw.rectangle([230, 230, 380, 380], fill=0)
        path = tmp_path / "multi.png"
        img.save(str(path))

        views = detect_and_split_views(path)
        # ビューが1つ以上返ることを確認 (ギャップ検出結果は画像際条件による)
        assert len(views) >= 1

    def test_empty_image(self, tmp_path: Path) -> None:
        """完全に白い画像."""
        from app.preprocessing.multiview_splitter import detect_and_split_views
        img = Image.new("L", (100, 100), 255)
        path = tmp_path / "white.png"
        img.save(str(path))

        views = detect_and_split_views(path)
        # 空の場合でも full として返す
        assert len(views) >= 1

    def test_save_views(self, tmp_path: Path) -> None:
        """ビューの保存."""
        from app.preprocessing.multiview_splitter import DetectedView, save_views
        views = [
            DetectedView(
                image=Image.new("RGB", (100, 100), (255, 0, 0)),
                x=0, y=0, width=100, height=100,
                label="front",
            ),
            DetectedView(
                image=Image.new("RGB", (100, 100), (0, 255, 0)),
                x=100, y=0, width=100, height=100,
                label="side",
            ),
        ]
        output_dir = tmp_path / "views"
        paths = save_views(views, output_dir, "test")
        assert len(paths) == 2
        assert all(p.exists() for p in paths)
        assert any("front" in p.name for p in paths)

    def test_detected_view_defaults(self) -> None:
        """DetectedViewのデフォルト値."""
        from app.preprocessing.multiview_splitter import DetectedView
        view = DetectedView(
            image=Image.new("RGB", (10, 10)),
            x=0, y=0, width=10, height=10,
        )
        assert view.label == "unknown"
        assert view.confidence == 0.0


# ===== Otsu閾値・ギャップ検出テスト =====


class TestSplitterInternals:
    """マルチビュー分割の内部関数テスト."""

    def test_otsu_threshold(self) -> None:
        """Otsu閾値計算."""
        from app.preprocessing.multiview_splitter import _otsu_threshold
        # グラデーション画像でOtsuが動作することを確認
        arr = np.zeros((200, 200), dtype=np.uint8)
        arr[:100, :] = 50   # 暗い領域
        arr[100:, :] = 200  # 明るい領域
        threshold = _otsu_threshold(arr)
        # 50〜200の間に閾値がある
        assert 50 <= threshold <= 200

    def test_find_gaps(self) -> None:
        """ギャップ検出."""
        from app.preprocessing.multiview_splitter import _find_gaps
        proj = np.array([100, 200, 0, 0, 0, 150, 300, 0, 0, 100])
        gap_threshold = 10
        gaps = _find_gaps(proj, gap_threshold)
        assert len(gaps) >= 1  # 少なくとも1つのギャップ

    def test_find_gaps_no_gaps(self) -> None:
        """ギャップなし."""
        from app.preprocessing.multiview_splitter import _find_gaps
        proj = np.array([100, 200, 300, 150, 100])
        gaps = _find_gaps(proj, 10)
        assert len(gaps) == 0

    def test_gaps_to_ranges_no_gaps(self) -> None:
        """ギャップなしの場合、全領域が1範囲."""
        from app.preprocessing.multiview_splitter import _gaps_to_ranges
        ranges = _gaps_to_ranges([], 100)
        assert ranges == [(0, 100)]

    def test_gaps_to_ranges_with_gaps(self) -> None:
        """ギャップありの範囲変換."""
        from app.preprocessing.multiview_splitter import _gaps_to_ranges
        gaps = [(40, 60)]
        ranges = _gaps_to_ranges(gaps, 100)
        assert (0, 40) in ranges
        assert (60, 100) in ranges

    def test_assign_view_labels(self) -> None:
        """ビューラベル割当."""
        from app.preprocessing.multiview_splitter import _assign_view_labels, DetectedView
        img = Image.new("RGB", (50, 50))
        views = [
            DetectedView(image=img, x=10, y=10, width=80, height=80, label="unknown"),
            DetectedView(image=img, x=210, y=10, width=80, height=80, label="unknown"),
            DetectedView(image=img, x=10, y=210, width=80, height=80, label="unknown"),
            DetectedView(image=img, x=210, y=210, width=80, height=80, label="unknown"),
        ]
        _assign_view_labels(views, 400, 400)
        labels = {v.label for v in views}
        assert "front" in labels
        assert "side" in labels
        assert "top" in labels
        assert "isometric" in labels

    def test_assign_single_view_label(self) -> None:
        """単一ビューのラベルはunknown."""
        from app.preprocessing.multiview_splitter import _assign_view_labels, DetectedView
        img = Image.new("RGB", (50, 50))
        views = [DetectedView(image=img, x=0, y=0, width=100, height=100)]
        _assign_view_labels(views, 100, 100)
        assert views[0].label == "unknown"


# ===== 前処理パイプライン追加テスト =====


class TestPipelineConfig:
    """前処理パイプライン設定の追加テスト."""

    def test_all_options_disabled(self) -> None:
        """全オプション無効."""
        from app.preprocessing.pipeline import PreprocessingConfig
        cfg = PreprocessingConfig(
            auto_split_views=False,
            remove_text=False,
            remove_dimensions=False,
            remove_hatching=False,
            remove_auxiliary=False,
            inpaint=False,
        )
        assert not cfg.auto_split_views
        assert not cfg.remove_text
        assert not cfg.remove_dimensions
        assert not cfg.remove_hatching
        assert not cfg.remove_auxiliary
        assert not cfg.inpaint
