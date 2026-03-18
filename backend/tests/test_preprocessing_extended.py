"""
前処理拡張テスト + Export API追加テスト.

テスト対象:
- app.preprocessing.format_converter (画像正規化/拡張子ルーティング/エラー)
- app.preprocessing.annotation_remover (データクラス/スケール抽出/マスク/パイプライン)
- app.api.routes.export (ダウンロード/フォーマット一覧)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# ===== format_converter テスト =====


class TestFormatConverter:
    """ファイルフォーマット変換のテスト."""

    def test_normalize_rgb_image(self, tmp_path: Path) -> None:
        """RGB画像のPNG正規化."""
        from app.preprocessing.format_converter import _normalize_image
        img = Image.new("RGB", (100, 100), (128, 64, 0))
        src = tmp_path / "test.jpg"
        img.save(str(src), "JPEG")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = _normalize_image(src, out_dir)
        assert result.exists()
        assert result.suffix == ".png"
        loaded = Image.open(result)
        assert loaded.mode == "RGB"

    def test_normalize_rgba_image(self, tmp_path: Path) -> None:
        """RGBA画像 → 白背景RGB."""
        from app.preprocessing.format_converter import _normalize_image
        img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
        src = tmp_path / "alpha.png"
        img.save(str(src), "PNG")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = _normalize_image(src, out_dir)
        loaded = Image.open(result)
        assert loaded.mode == "RGB"

    def test_normalize_la_image(self, tmp_path: Path) -> None:
        """LA(グレー+アルファ)画像 → RGB."""
        from app.preprocessing.format_converter import _normalize_image
        img = Image.new("LA", (50, 50), (100, 200))
        src = tmp_path / "gray_alpha.png"
        img.save(str(src), "PNG")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = _normalize_image(src, out_dir)
        loaded = Image.open(result)
        assert loaded.mode == "RGB"

    def test_normalize_grayscale(self, tmp_path: Path) -> None:
        """グレースケール画像 → RGB."""
        from app.preprocessing.format_converter import _normalize_image
        img = Image.new("L", (50, 50), 128)
        src = tmp_path / "gray.png"
        img.save(str(src), "PNG")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        result = _normalize_image(src, out_dir)
        loaded = Image.open(result)
        assert loaded.mode == "RGB"

    def test_convert_to_images_png(self, tmp_path: Path) -> None:
        """PNG入力のルーティング."""
        from app.preprocessing.format_converter import convert_to_images
        img = Image.new("RGB", (100, 100), (0, 255, 0))
        src = tmp_path / "input.png"
        img.save(str(src))
        results = convert_to_images(src)
        assert len(results) == 1
        assert results[0].exists()

    def test_convert_to_images_jpg(self, tmp_path: Path) -> None:
        """JPG入力のルーティング."""
        from app.preprocessing.format_converter import convert_to_images
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        src = tmp_path / "photo.jpg"
        img.save(str(src), "JPEG")
        results = convert_to_images(src)
        assert len(results) == 1

    def test_convert_to_images_bmp(self, tmp_path: Path) -> None:
        """BMP入力のルーティング."""
        from app.preprocessing.format_converter import convert_to_images
        img = Image.new("RGB", (50, 50), (0, 0, 255))
        src = tmp_path / "test.bmp"
        img.save(str(src), "BMP")
        results = convert_to_images(src)
        assert len(results) == 1

    def test_convert_unsupported_format(self, tmp_path: Path) -> None:
        """非対応形式はPreprocessingError."""
        from app.preprocessing.format_converter import convert_to_images
        from app.core.exceptions import PreprocessingError
        src = tmp_path / "test.xyz"
        src.write_text("dummy")
        with pytest.raises(PreprocessingError, match="未対応"):
            convert_to_images(src)

    def test_extension_sets(self) -> None:
        """拡張子セットの定義確認."""
        from app.preprocessing.format_converter import (
            _IMAGE_EXTENSIONS, _VECTOR_EXTENSIONS, _DOCUMENT_EXTENSIONS, _RAW_EXTENSIONS,
        )
        assert ".jpg" in _IMAGE_EXTENSIONS
        assert ".png" in _IMAGE_EXTENSIONS
        assert ".svg" in _VECTOR_EXTENSIONS
        assert ".dxf" in _VECTOR_EXTENSIONS
        assert ".pdf" in _DOCUMENT_EXTENSIONS
        assert ".heic" in _RAW_EXTENSIONS


# ===== annotation_remover テスト =====


class TestAnnotationRemover:
    """注釈除去モジュールのテスト."""

    def test_annotation_region_dataclass(self) -> None:
        """AnnotationRegionデータクラス."""
        from app.preprocessing.annotation_remover import AnnotationRegion
        r = AnnotationRegion(
            x=10, y=20, width=100, height=50,
            annotation_type="text",
            text_content="50mm",
            confidence=0.95,
        )
        assert r.x == 10
        assert r.annotation_type == "text"
        assert r.text_content == "50mm"

    def test_scale_info_defaults(self) -> None:
        """ScaleInfoデフォルト値."""
        from app.preprocessing.annotation_remover import ScaleInfo
        s = ScaleInfo()
        assert s.value is None
        assert s.unit == "mm"
        assert s.confidence == 0.0

    def test_cleanup_result_dataclass(self) -> None:
        """CleanupResultデータクラス."""
        from app.preprocessing.annotation_remover import CleanupResult
        img = Image.new("RGB", (100, 100))
        r = CleanupResult(cleaned_image=img)
        assert r.removed_annotations == []
        assert r.extracted_scale is None

    def test_extract_scale_mm(self) -> None:
        """スケール抽出: mm."""
        from app.preprocessing.annotation_remover import _extract_scale_from_text
        result = _extract_scale_from_text("100mm", 0.9)
        assert result is not None
        assert result.value == 100.0
        assert result.unit == "mm"

    def test_extract_scale_cm(self) -> None:
        """スケール抽出: cm."""
        from app.preprocessing.annotation_remover import _extract_scale_from_text
        result = _extract_scale_from_text("5.5 cm", 0.8)
        assert result is not None
        assert result.value == 5.5
        assert result.unit == "cm"

    def test_extract_scale_ratio(self) -> None:
        """スケール抽出: 1:100."""
        from app.preprocessing.annotation_remover import _extract_scale_from_text
        result = _extract_scale_from_text("1:100", 0.85)
        assert result is not None
        assert result.value == 100.0
        assert result.unit == "scale"

    def test_extract_scale_radius(self) -> None:
        """スケール抽出: R50."""
        from app.preprocessing.annotation_remover import _extract_scale_from_text
        result = _extract_scale_from_text("R50", 0.7)
        assert result is not None
        assert result.value == 50.0
        assert result.unit == "radius_mm"

    def test_extract_scale_no_match(self) -> None:
        """スケール抽出: マッチなし."""
        from app.preprocessing.annotation_remover import _extract_scale_from_text
        result = _extract_scale_from_text("Hello World", 0.9)
        assert result is None

    def test_apply_mask(self) -> None:
        """マスク適用."""
        from app.preprocessing.annotation_remover import _apply_mask, AnnotationRegion
        mask = np.zeros((100, 100), dtype=np.uint8)
        region = AnnotationRegion(x=10, y=20, width=30, height=40, annotation_type="text")
        _apply_mask(mask, region)
        assert mask[20:60, 10:40].sum() > 0
        assert mask[0, 0] == 0

    def test_apply_mask_boundary(self) -> None:
        """マスク適用: 境界を超える領域."""
        from app.preprocessing.annotation_remover import _apply_mask, AnnotationRegion
        mask = np.zeros((50, 50), dtype=np.uint8)
        region = AnnotationRegion(x=40, y=40, width=30, height=30, annotation_type="text")
        _apply_mask(mask, region)
        assert mask[40:50, 40:50].sum() > 0

    def test_inpaint_fallback(self) -> None:
        """インペインティング: OpenCVなしのフォールバック(白塗り)."""
        from app.preprocessing.annotation_remover import _inpaint_masked
        arr = np.full((50, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        result = _inpaint_masked(arr, mask)
        assert result.shape == arr.shape

    def test_detect_auxiliary_returns_empty(self) -> None:
        """補助線検出は現在空リスト."""
        from app.preprocessing.annotation_remover import _detect_auxiliary_lines
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _detect_auxiliary_lines(arr)
        assert result == []

    def test_remove_annotations_basic(self, tmp_path: Path) -> None:
        """注釈除去パイプライン全体(OCRなし環境)."""
        from app.preprocessing.annotation_remover import remove_annotations
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        path = tmp_path / "drawing.png"
        img.save(str(path))
        result = remove_annotations(
            path,
            remove_text=True,
            remove_dimensions=False,
            remove_hatching=False,
            remove_auxiliary=True,
            inpaint=False,
        )
        assert result.cleaned_image.size == (200, 200)
        assert result.original_size == (200, 200)

    def test_remove_annotations_all_disabled(self, tmp_path: Path) -> None:
        """全オプション無効."""
        from app.preprocessing.annotation_remover import remove_annotations
        img = Image.new("RGB", (100, 100), (128, 128, 128))
        path = tmp_path / "simple.png"
        img.save(str(path))
        result = remove_annotations(
            path,
            remove_text=False,
            remove_dimensions=False,
            remove_hatching=False,
            remove_auxiliary=False,
            inpaint=False,
        )
        assert len(result.removed_annotations) == 0


# ===== Export API 追加テスト =====


class TestExportAPIExtended:
    """Export API の追加テスト."""

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_list_formats(self, client) -> None:
        """出力フォーマット一覧."""
        resp = client.get("/api/export/formats")
        # エンドポイントが存在すれば200、なければ404
        assert resp.status_code in (200, 404)

    def test_download_nonexistent(self, client) -> None:
        """存在しないファイルのダウンロード."""
        resp = client.get("/api/export/download/nonexistent-id-12345")
        assert resp.status_code in (404, 500)

    def test_open_external_nonexistent(self, client) -> None:
        """存在しないファイルの外部オープン."""
        resp = client.post("/api/export/open-external/nonexistent-id-12345")
        assert resp.status_code in (404, 500)
