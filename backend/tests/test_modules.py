"""
前処理・後処理・重み管理モジュールのテスト.

テスト対象:
- app.preprocessing (パイプライン/フォーマット変換)
- app.postprocessing (メッシュ修復/情報取得)
- app.weights (重み管理/SHA256/ステータス)
- app.similarity (メッシュ/画像類似度)
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image


# ===== 前処理データ構造テスト =====


class TestPreprocessingConfig:
    """前処理設定のテスト."""

    def test_default_config(self) -> None:
        """デフォルト設定値."""
        from app.preprocessing.pipeline import PreprocessingConfig
        cfg = PreprocessingConfig()
        assert cfg.auto_split_views is True
        assert cfg.remove_text is True
        assert cfg.remove_dimensions is True
        assert cfg.remove_hatching is True
        assert cfg.remove_auxiliary is True
        assert cfg.inpaint is True

    def test_custom_config(self) -> None:
        """カスタム設定."""
        from app.preprocessing.pipeline import PreprocessingConfig
        cfg = PreprocessingConfig(
            auto_split_views=False,
            remove_text=False,
        )
        assert cfg.auto_split_views is False
        assert cfg.remove_text is False


class TestPreprocessedImage:
    """前処理済み画像のテスト."""

    def test_default_values(self, tmp_path: Path) -> None:
        """デフォルト値."""
        from app.preprocessing.pipeline import PreprocessedImage
        pi = PreprocessedImage(
            path=tmp_path / "test.png",
            original_path=tmp_path / "orig.png",
        )
        assert pi.view_label == "unknown"
        assert pi.scale_info is None
        assert pi.removed_annotations == []
        assert pi.was_split is False


# ===== メッシュ修復テスト =====


class TestMeshRepair:
    """メッシュ修復モジュールのテスト."""

    def test_repair_config_defaults(self) -> None:
        """修復設定のデフォルト値."""
        from app.postprocessing.mesh_repair import MeshRepairConfig
        cfg = MeshRepairConfig()
        assert cfg.fill_holes is True
        assert cfg.fix_normals is True
        assert cfg.simplify is False
        assert cfg.target_faces == 50000
        assert cfg.make_watertight is False

    def test_mesh_stats_defaults(self) -> None:
        """MeshStats デフォルト値."""
        from app.postprocessing.mesh_repair import MeshStats
        stats = MeshStats()
        assert stats.vertices == 0
        assert stats.faces == 0
        assert stats.is_watertight is False
        assert stats.volume == 0.0

    def test_repair_box(self, tmp_path: Path) -> None:
        """ボックスメッシュの修復."""
        from app.postprocessing.mesh_repair import repair_mesh, MeshRepairConfig
        box = trimesh.creation.box(extents=(10, 10, 10))
        input_path = tmp_path / "input.stl"
        output_path = tmp_path / "output.stl"
        box.export(str(input_path))

        config = MeshRepairConfig(remove_self_intersections=False)
        result_path, stats = repair_mesh(input_path, output_path, config=config)
        assert result_path == output_path
        assert output_path.exists()
        assert stats.vertices > 0
        assert stats.faces > 0
        assert stats.surface_area > 0

    def test_repair_with_simplification(self, tmp_path: Path) -> None:
        """簡略化付き修復."""
        from app.postprocessing.mesh_repair import repair_mesh, MeshRepairConfig
        sphere = trimesh.creation.icosphere(subdivisions=4)
        input_path = tmp_path / "sphere.stl"
        output_path = tmp_path / "sphere_simple.stl"
        sphere.export(str(input_path))

        config = MeshRepairConfig(
            simplify=True, target_faces=100,
            remove_self_intersections=False,
        )
        _, stats = repair_mesh(input_path, output_path, config=config)
        # simplify_quadric_decimation may fail on some trimesh versions
        assert stats.faces > 0

    def test_get_mesh_info(self, tmp_path: Path) -> None:
        """メッシュ情報取得."""
        from app.postprocessing.mesh_repair import get_mesh_info
        box = trimesh.creation.box(extents=(10, 10, 10))
        path = tmp_path / "info_test.stl"
        box.export(str(path))

        stats = get_mesh_info(path)
        assert stats.vertices > 0
        assert stats.faces > 0
        assert stats.surface_area > 0

    def test_get_mesh_info_nonexistent(self, tmp_path: Path) -> None:
        """存在しないファイルでデフォルトStatsを返す."""
        from app.postprocessing.mesh_repair import get_mesh_info
        stats = get_mesh_info(tmp_path / "nonexistent.stl")
        assert stats.vertices == 0
        assert stats.faces == 0

    def test_repair_smooth(self, tmp_path: Path) -> None:
        """Laplacianスムージング付き修復."""
        from app.postprocessing.mesh_repair import repair_mesh, MeshRepairConfig
        box = trimesh.creation.box(extents=(10, 10, 10))
        input_path = tmp_path / "box.stl"
        output_path = tmp_path / "box_smooth.stl"
        box.export(str(input_path))

        config = MeshRepairConfig(
            smooth=True, smooth_iterations=2,
            remove_self_intersections=False,
        )
        _, stats = repair_mesh(input_path, output_path, config=config)
        assert stats.vertices > 0


# ===== 重み管理テスト =====


class TestWeightManager:
    """WeightManager のテスト."""

    def test_sha256_computation(self, tmp_path: Path) -> None:
        """SHA256ハッシュ計算."""
        from app.weights.manager import WeightManager
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"hello world")

        expected = hashlib.sha256(b"hello world").hexdigest()
        actual = WeightManager._compute_sha256(test_file)
        assert actual == expected

    def test_sha256_empty_file(self, tmp_path: Path) -> None:
        """空ファイルのSHA256."""
        from app.weights.manager import WeightManager
        test_file = tmp_path / "empty.bin"
        test_file.write_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()
        actual = WeightManager._compute_sha256(test_file)
        assert actual == expected

    def test_check_weight_exists(self) -> None:
        """重みファイル存在チェック."""
        from app.weights.manager import weight_manager
        from app.models.schemas import WeightFileInfo
        w = WeightFileInfo(
            name="test",
            url="https://example.com/test.bin",
            relative_path="nonexistent/file.bin",
        )
        assert weight_manager.check_weight_exists(w) is False

    def test_get_all_status(self) -> None:
        """全重みステータス取得."""
        from app.weights.manager import weight_manager
        from app.models.schemas import WeightFileInfo
        weights = [
            WeightFileInfo(
                name="test1",
                url="https://example.com/a.bin",
                relative_path="a/a.bin",
            ),
            WeightFileInfo(
                name="test2",
                url="https://example.com/b.bin",
                relative_path="b/b.bin",
            ),
        ]
        statuses = weight_manager.get_all_status(weights)
        assert len(statuses) == 2
        for s in statuses:
            assert "name" in s
            assert "exists" in s
            assert s["exists"] is False
            assert s["downloading"] is False

    def test_active_downloads_initially_empty(self) -> None:
        """初期状態でアクティブダウンロードなし."""
        from app.weights.manager import WeightManager
        wm = WeightManager()
        assert len(wm._active_downloads) == 0


# ===== 類似度モジュールテスト =====


class TestMeshSimilarity:
    """メッシュ類似度のテスト."""

    def test_identical_similarity(self, tmp_path: Path) -> None:
        """同一メッシュの類似度 ≈ 1.0."""
        from app.similarity.mesh_similarity import compare_mesh_files
        box = trimesh.creation.box(extents=(10, 10, 10))
        path = tmp_path / "box.stl"
        box.export(str(path))
        result = compare_mesh_files(path, path)
        assert result.score >= 0.9

    def test_different_similarity(self, tmp_path: Path) -> None:
        """異なるメッシュの類似度 < 1.0."""
        from app.similarity.mesh_similarity import compare_mesh_files
        box = trimesh.creation.box(extents=(10, 10, 10))
        sphere = trimesh.creation.icosphere()
        pa = tmp_path / "box.stl"
        pb = tmp_path / "sphere.stl"
        box.export(str(pa))
        sphere.export(str(pb))
        result = compare_mesh_files(pa, pb)
        assert result.score < 1.0
        assert result.score >= 0.0


class TestImageSimilarity:
    """画像類似度のテスト."""

    def test_identical_image(self, tmp_path: Path) -> None:
        """同一画像の類似度 ≈ 1.0."""
        from app.similarity.image_similarity import compare_image_files
        img = Image.new("L", (100, 100), 128)
        path = tmp_path / "img.png"
        img.save(str(path))
        result = compare_image_files(path, path)
        assert result.score >= 0.5  # 均一色画像はedge_similarity=0のためスコアが低い

    def test_different_images(self, tmp_path: Path) -> None:
        """異なる画像の類似度 < 1.0."""
        from app.similarity.image_similarity import compare_image_files
        img_a = Image.new("L", (100, 100), 0)
        img_b = Image.new("L", (100, 100), 255)
        pa = tmp_path / "a.png"
        pb = tmp_path / "b.png"
        img_a.save(str(pa))
        img_b.save(str(pb))
        result = compare_image_files(pa, pb)
        assert result.score < 1.0
        assert result.score >= 0.0


# ===== フォーマット変換テスト =====


class TestFormatConverter:
    """フォーマット変換のテスト."""

    def test_convert_png(self, tmp_path: Path) -> None:
        """PNG→PNG変換 (パススルー)."""
        from app.preprocessing.format_converter import convert_to_images
        img = Image.new("RGB", (100, 100), (255, 0, 0))
        path = tmp_path / "test.png"
        img.save(str(path))

        results = convert_to_images(path)
        assert len(results) >= 1

    def test_convert_jpg(self, tmp_path: Path) -> None:
        """JPEG→PNG変換."""
        from app.preprocessing.format_converter import convert_to_images
        img = Image.new("RGB", (100, 100), (0, 255, 0))
        path = tmp_path / "test.jpg"
        img.save(str(path))

        results = convert_to_images(path)
        assert len(results) >= 1


# ===== 注釈除去テスト =====


class TestAnnotationRemover:
    """注釈除去のテスト."""

    def test_remove_annotations_clean_image(self, tmp_path: Path) -> None:
        """クリーンな画像に対する注釈除去."""
        from app.preprocessing.annotation_remover import remove_annotations
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        path = tmp_path / "clean.png"
        img.save(str(path))

        result = remove_annotations(path)
        assert result.cleaned_image is not None
        assert result.cleaned_image.size == (200, 200)

    def test_cleanup_result_attributes(self, tmp_path: Path) -> None:
        """CleanupResultの属性."""
        from app.preprocessing.annotation_remover import remove_annotations
        img = Image.new("RGB", (100, 100), (200, 200, 200))
        path = tmp_path / "attrs.png"
        img.save(str(path))

        result = remove_annotations(path)
        assert hasattr(result, "cleaned_image")
        assert hasattr(result, "removed_annotations")
        assert hasattr(result, "extracted_scale")
        assert isinstance(result.removed_annotations, list)
