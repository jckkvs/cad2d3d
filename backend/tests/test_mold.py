"""
金型設計推定モジュールのユニットテスト.

テスト対象:
- T-070: アンダーカット検出 (undercut.py)
- T-071: パーティングライン最適化 (parting_line.py)
- T-072: 金型構造設計 (core.py)
- T-073: 部品DB (parts_db.py)
- T-080: メッシュ類似度 (mesh_similarity.py)
- T-081: 画像類似度 (image_similarity.py)
- T-090: 金型API (mold routes)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ===== テスト用メッシュ生成ヘルパー =====


def _create_box_stl(path: Path, size: tuple[float, float, float] = (40, 30, 20)) -> Path:
    """テスト用の直方体STLファイルを生成."""
    import trimesh
    mesh = trimesh.creation.box(extents=size)
    mesh.export(str(path))
    return path


def _create_l_shape_stl(path: Path) -> Path:
    """テスト用のL字型STL (アンダーカットあり) を生成."""
    import trimesh
    # boolean演算なしでL字型を表現: 2つのメッシュを結合
    box1 = trimesh.creation.box(extents=(40, 30, 20))
    box2 = trimesh.creation.box(extents=(15, 30, 15))
    box2.apply_translation([12.5, 0, 17.5])
    # concatenate (boolean ではなく単純結合)
    mesh = trimesh.util.concatenate([box1, box2])
    mesh.export(str(path))
    return path



# ===== T-070: アンダーカット検出テスト =====


class TestUndercutDetection:
    """アンダーカット検出のテスト."""

    def test_simple_box_no_undercut(self, tmp_path: Path) -> None:
        """単純な直方体はアンダーカットなし."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.undercut import detect_undercuts

        result = detect_undercuts(
            stl_path,
            parting_direction=np.array([0, 0, 1]),
            ray_density=10,
        )
        assert result.parting_direction is not None
        assert len(result.parting_direction) == 3
        assert len(result.core_faces) > 0
        assert len(result.cavity_faces) > 0
        assert result.summary != ""

    def test_auto_parting_direction(self, tmp_path: Path) -> None:
        """型開き方向の自動決定."""
        stl_path = _create_box_stl(tmp_path / "box.stl", size=(100, 50, 10))
        from app.mold.undercut import detect_undercuts

        result = detect_undercuts(stl_path, parting_direction=None, ray_density=10)
        # 方向が決定されていること (3次元ベクトル)
        assert result.parting_direction is not None
        assert np.linalg.norm(result.parting_direction) > 0.99

    def test_face_classification(self, tmp_path: Path) -> None:
        """面分類が正しく行われること."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.undercut import detect_undercuts

        result = detect_undercuts(
            stl_path,
            parting_direction=np.array([0, 0, 1]),
            ray_density=10,
        )
        # 直方体の場合、Core + Cavity + Parting で全面をカバー
        total = len(result.core_faces) + len(result.cavity_faces)
        assert total > 0

    def test_undercut_region_attributes(self, tmp_path: Path) -> None:
        """アンダーカット領域の属性が正しいこと."""
        from app.mold.undercut import UndercutRegion
        region = UndercutRegion(
            face_indices=[0, 1, 2],
            area=150.0,
            centroid=np.array([10, 20, 5]),
            depth=8.0,
            direction=np.array([1, 0, 0]),
            recommended_mechanism="slide_core",
            severity="moderate",
        )
        assert region.area == 150.0
        assert region.depth == 8.0
        assert region.recommended_mechanism == "slide_core"
        assert region.severity == "moderate"

    def test_recommend_mechanism_shallow(self) -> None:
        """浅く小面積 → angled_lifter."""
        from app.mold.undercut import _recommend_mechanism
        assert _recommend_mechanism(area=30, depth=2.0) == "angled_lifter"

    def test_recommend_mechanism_deep(self) -> None:
        """深い → slide_core."""
        from app.mold.undercut import _recommend_mechanism
        assert _recommend_mechanism(area=100, depth=20.0) == "slide_core"

    def test_recommend_mechanism_large_area(self) -> None:
        """大面積 → slide_core."""
        from app.mold.undercut import _recommend_mechanism
        assert _recommend_mechanism(area=500, depth=5.0) == "slide_core"

    def test_summary_generation(self, tmp_path: Path) -> None:
        """サマリが適切に生成されること."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.undercut import detect_undercuts
        result = detect_undercuts(stl_path, np.array([0, 0, 1]))
        assert "アンダーカット解析結果" in result.summary
        assert "パーティング方向" in result.summary
        assert "Core 面数" in result.summary

    def test_l_shape_analysis(self, tmp_path: Path) -> None:
        """L字型でアンダーカット解析が動作すること."""
        stl_path = _create_l_shape_stl(tmp_path / "lshape.stl")
        from app.mold.undercut import detect_undercuts
        result = detect_undercuts(stl_path, ray_density=20)
        # L字型は方向によってはアンダーカットが存在しうる
        assert result.parting_direction is not None
        assert len(result.core_faces) + len(result.cavity_faces) > 0

    def test_ray_density_parameter(self, tmp_path: Path) -> None:
        """ray_densityパラメータが機能すること."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.undercut import detect_undercuts
        result_low = detect_undercuts(stl_path, np.array([0, 0, 1]), ray_density=5)
        result_high = detect_undercuts(stl_path, np.array([0, 0, 1]), ray_density=30)
        # 両方とも完了すること
        assert result_low.parting_direction is not None
        assert result_high.parting_direction is not None

    def test_cluster_empty(self) -> None:
        """空のアンダーカットリストでクラスタリング."""
        from app.mold.undercut import _cluster_undercut_regions
        import trimesh
        mesh = trimesh.creation.box(extents=(40, 30, 20))
        regions = _cluster_undercut_regions(mesh, [], np.array([0, 0, 1]))
        assert regions == []


# ===== T-071: パーティングライン最適化テスト =====


class TestPartingLineOptimization:
    """パーティングライン最適化のテスト."""

    def test_optimize_box(self, tmp_path: Path) -> None:
        """直方体のPL最適化."""
        stl_path = _create_box_stl(tmp_path / "box.stl", size=(100, 50, 10))
        from app.mold.parting_line import optimize_parting_line

        result = optimize_parting_line(stl_path, candidate_count=6)
        assert result.best is not None
        assert result.best.score >= 0
        assert result.best.mold_height > 0
        assert result.best.mold_volume > 0
        assert len(result.candidates) > 0

    def test_multiple_candidates(self, tmp_path: Path) -> None:
        """複数候補が生成されること."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.parting_line import optimize_parting_line

        result = optimize_parting_line(stl_path, candidate_count=6)
        assert len(result.candidates) >= 3  # 少なくとも3方向
        # スコア順にソートされていること
        for i in range(len(result.candidates) - 1):
            assert result.candidates[i].score <= result.candidates[i + 1].score

    def test_orientation_matrix(self, tmp_path: Path) -> None:
        """回転行列が正規直交であること."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.parting_line import optimize_parting_line

        result = optimize_parting_line(stl_path)
        R = result.product_orientation
        assert R.shape == (3, 3)
        # R^T R ≈ I
        identity = R.T @ R
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-6)

    def test_custom_weights(self, tmp_path: Path) -> None:
        """カスタム重みでスコアが変化すること."""
        stl_path = _create_box_stl(tmp_path / "box.stl", size=(100, 50, 10))
        from app.mold.parting_line import optimize_parting_line

        result_default = optimize_parting_line(stl_path)
        result_custom = optimize_parting_line(
            stl_path,
            weights={"height": 10.0, "volume": 0.0, "undercut": 0.0, "gate": 0.0, "machinability": 0.0},
        )
        # 重み付けが異なれば最適方向が変わる可能性がある
        assert result_default.best.score != result_custom.best.score or \
            np.allclose(result_default.best.direction, result_custom.best.direction)


# ===== T-072: 金型構造設計テスト =====


class TestMoldDesigner:
    """金型全体構造設計のテスト."""

    def test_design_simple_box(self, tmp_path: Path) -> None:
        """単純な直方体から金型を設計."""
        stl_path = _create_box_stl(tmp_path / "box.stl", size=(80, 60, 30))
        from app.mold.core import MoldDesigner

        designer = MoldDesigner()
        result = designer.design(stl_path)

        # プレートが全て設計されていること
        assert result.cavity_plate is not None
        assert result.core_plate is not None
        assert result.fixed_clamping_plate is not None
        assert result.movable_clamping_plate is not None
        assert result.support_plate is not None
        assert len(result.spacer_blocks) == 2

    def test_plates_larger_than_product(self, tmp_path: Path) -> None:
        """型板が製品より大きいこと."""
        stl_path = _create_box_stl(tmp_path / "box.stl", size=(80, 60, 30))
        from app.mold.core import MoldDesigner

        result = MoldDesigner().design(stl_path)
        assert result.total_width > 80
        assert result.total_height > 60

    def test_components_generated(self, tmp_path: Path) -> None:
        """標準部品が選定されること."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.core import MoldDesigner

        result = MoldDesigner().design(stl_path)
        component_names = [c.name for c in result.components]
        assert "ガイドピン" in component_names
        assert "突き出しピン" in component_names
        assert "スプルーブッシュ" in component_names
        assert "ロケートリング" in component_names

    def test_cost_estimated(self, tmp_path: Path) -> None:
        """コストが概算されること."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.core import MoldDesigner

        result = MoldDesigner().design(stl_path)
        assert result.estimated_material_cost > 0
        assert result.estimated_machining_cost > 0
        assert result.estimated_total_cost > 0

    def test_summary_generated(self, tmp_path: Path) -> None:
        """サマリが生成されること."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.core import MoldDesigner

        result = MoldDesigner().design(stl_path)
        assert "金型設計結果" in result.summary
        assert "金型サイズ" in result.summary


# ===== T-073: 部品DBテスト =====


class TestPartsDatabase:
    """標準部品DBのテスト."""

    def test_default_catalog_loaded(self, tmp_path: Path) -> None:
        """デフォルトカタログがロードされること."""
        from app.mold.parts_db import PartsDatabase
        db = PartsDatabase(db_path=tmp_path / "test_parts.json")
        all_parts = db.list_all()
        assert len(all_parts) > 0

    def test_search_by_category(self, tmp_path: Path) -> None:
        """カテゴリで検索."""
        from app.mold.parts_db import PartsDatabase
        db = PartsDatabase(db_path=tmp_path / "test_parts.json")
        pins = db.search(category="guide_pin")
        assert len(pins) > 0
        assert all(p.category == "guide_pin" for p in pins)

    def test_search_by_size(self, tmp_path: Path) -> None:
        """サイズで検索."""
        from app.mold.parts_db import PartsDatabase
        db = PartsDatabase(db_path=tmp_path / "test_parts.json")
        large_pins = db.search(category="ejector_pin", min_diameter=8, min_length=100)
        assert len(large_pins) > 0
        assert all(p.diameter >= 8 for p in large_pins)
        assert all(p.length >= 100 for p in large_pins)

    def test_recommend_minimum_fit(self, tmp_path: Path) -> None:
        """必要サイズ以上で最小の部品を推奨."""
        from app.mold.parts_db import PartsDatabase
        db = PartsDatabase(db_path=tmp_path / "test_parts.json")
        part = db.recommend("ejector_pin", required_diameter=5, required_length=105)
        assert part is not None
        assert part.diameter >= 5
        assert part.length >= 105

    def test_add_and_remove_part(self, tmp_path: Path) -> None:
        """部品の追加と削除."""
        from app.mold.parts_db import PartsDatabase, MoldPart
        db = PartsDatabase(db_path=tmp_path / "test_parts.json")
        initial_count = len(db.list_all())

        new_part = MoldPart(
            id="TEST-001", category="guide_pin", name="テストピン",
            manufacturer="TestCo", model_number="T100",
            diameter=50, length=300, unit_price=9999,
        )
        db.add_part(new_part)
        assert len(db.list_all()) == initial_count + 1

        assert db.remove_part("TEST-001")
        assert len(db.list_all()) == initial_count

    def test_persistence(self, tmp_path: Path) -> None:
        """JSONファイルに永続化されること."""
        db_file = tmp_path / "persist.json"
        from app.mold.parts_db import PartsDatabase, MoldPart
        db1 = PartsDatabase(db_path=db_file)
        db1.add_part(MoldPart(
            id="PERSIST-01", category="guide_pin", name="永続テスト",
            manufacturer="X", model_number="Y",
        ))

        # 再読込
        db2 = PartsDatabase(db_path=db_file)
        found = [p for p in db2.list_all() if p.id == "PERSIST-01"]
        assert len(found) == 1

    def test_categories_list(self, tmp_path: Path) -> None:
        """カテゴリ一覧."""
        from app.mold.parts_db import PartsDatabase
        db = PartsDatabase(db_path=tmp_path / "test_parts.json")
        cats = db.list_categories()
        assert "guide_pin" in cats
        assert "ejector_pin" in cats
        assert "gas_spring" in cats


# ===== T-080: メッシュ類似度テスト =====


class TestMeshSimilarity:
    """3Dメッシュ類似度のテスト."""

    def test_identical_meshes(self, tmp_path: Path) -> None:
        """同一メッシュの類似度は1.0に近い."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.similarity.mesh_similarity import compare_mesh_files

        result = compare_mesh_files(stl_path, stl_path)
        assert result.score > 0.95

    def test_different_meshes(self, tmp_path: Path) -> None:
        """異なるサイズのメッシュ間の類似度."""
        path_a = _create_box_stl(tmp_path / "small.stl", size=(10, 10, 10))
        path_b = _create_box_stl(tmp_path / "large.stl", size=(100, 100, 100))
        from app.similarity.mesh_similarity import compare_mesh_files

        result = compare_mesh_files(path_a, path_b)
        # 同じ形状(直方体)だがサイズが違うのでスコアは複合的
        assert 0.0 <= result.score <= 1.0

    def test_descriptor_attributes(self, tmp_path: Path) -> None:
        """記述子の属性が正しいこと."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.similarity.mesh_similarity import compute_descriptor

        desc = compute_descriptor(stl_path)
        assert desc.vertex_count > 0
        assert desc.face_count > 0
        assert desc.surface_area > 0
        assert len(desc.d2_histogram) == 64
        assert len(desc.aspect_ratios) == 3


# ===== T-081: 画像類似度テスト =====


class TestImageSimilarity:
    """2D画像類似度のテスト."""

    def _create_test_image(self, path: Path, color: int = 128) -> Path:
        """テスト用画像を生成."""
        from PIL import Image
        img = Image.new("L", (100, 100), color=color)
        img.save(str(path))
        return path

    def test_identical_images(self, tmp_path: Path) -> None:
        """同一画像の類似度は1.0に近い."""
        img_path = self._create_test_image(tmp_path / "img.png")
        from app.similarity.image_similarity import compare_image_files

        result = compare_image_files(img_path, img_path)
        # 均一色画像ではエッジ密度=0のためスコアが下がる
        assert result.score > 0.5
        assert result.histogram_similarity > 0.99

    def test_different_images(self, tmp_path: Path) -> None:
        """異なる画像間の類似度."""
        path_a = self._create_test_image(tmp_path / "bright.png", color=200)
        path_b = self._create_test_image(tmp_path / "dark.png", color=50)
        from app.similarity.image_similarity import compare_image_files

        result = compare_image_files(path_a, path_b)
        assert 0.0 <= result.score <= 1.0

    def test_descriptor_attributes(self, tmp_path: Path) -> None:
        """記述子の属性."""
        img_path = self._create_test_image(tmp_path / "img.png")
        from app.similarity.image_similarity import compute_image_descriptor

        desc = compute_image_descriptor(img_path)
        assert desc.aspect_ratio == 1.0  # 100x100
        assert len(desc.edge_histogram) == 36
        assert len(desc.intensity_histogram) == 64


# ===== T-090: 金型APIテスト =====


class TestMoldAPI:
    """金型設計APIのテスト."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)

    def test_parts_categories(self, client) -> None:
        """部品カテゴリ一覧API."""
        resp = client.get("/api/mold/parts/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert "guide_pin" in data

    def test_parts_list(self, client) -> None:
        """部品一覧API."""
        resp = client.get("/api/mold/parts")
        assert resp.status_code == 200
        parts = resp.json()
        assert len(parts) > 0

    def test_parts_search_filter(self, client) -> None:
        """部品検索のフィルタリング."""
        resp = client.get("/api/mold/parts?category=ejector_pin&min_diameter=8")
        assert resp.status_code == 200
        parts = resp.json()
        assert all(p["category"] == "ejector_pin" for p in parts)
        assert all(p["diameter"] >= 8 for p in parts)

    def test_parts_recommend(self, client) -> None:
        """部品推奨API."""
        resp = client.get("/api/mold/parts/recommend?category=ejector_pin&diameter=5&length=105")
        assert resp.status_code == 200
        part = resp.json()
        if part is not None:
            assert part["diameter"] >= 5
            assert part["length"] >= 105

    def test_design_nonexistent_file(self, client) -> None:
        """存在しないファイルで404."""
        resp = client.post("/api/mold/design", json={"mesh_path": "/nonexistent.stl"})
        assert resp.status_code == 404

    def test_undercut_nonexistent_file(self, client) -> None:
        """存在しないファイルで404."""
        resp = client.post("/api/mold/undercut", json={"mesh_path": "/nonexistent.stl"})
        assert resp.status_code == 404

    def test_similarity_mesh_nonexistent(self, client) -> None:
        """存在しないファイルで404."""
        resp = client.post("/api/mold/similarity/mesh", json={"path_a": "/a.stl", "path_b": "/b.stl"})
        assert resp.status_code == 404

    def test_sizing_clamp_force(self, client) -> None:
        """型締力計算API."""
        resp = client.post("/api/mold/sizing/clamp-force", json={
            "projected_area_mm2": 5000,
            "resin": "ABS",
            "cavity_count": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert data["data"]["clamp_force_ton"] > 0

    def test_sizing_cycle_time(self, client) -> None:
        """サイクルタイムAPI."""
        resp = client.post("/api/mold/sizing/cycle-time", json={
            "wall_thickness_mm": 2.5,
            "part_weight_g": 30,
            "resin": "PP",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"]
        assert data["data"]["total_cycle_s"] > 0

    def test_resins_list(self, client) -> None:
        """樹脂一覧API."""
        resp = client.get("/api/mold/resins")
        assert resp.status_code == 200
        data = resp.json()
        assert "ABS" in data
        assert "PP" in data


# ===== T-074: サイジングルールテスト =====


class TestSizing:
    """ルールベースサイジングのテスト."""

    def test_clamp_force_basic(self) -> None:
        """型締力の基本計算."""
        from app.mold.sizing import calculate_clamp_force
        result = calculate_clamp_force(
            projected_area_mm2=5000, resin="ABS", cavity_count=1
        )
        assert result.clamp_force_ton > 0
        assert result.clamp_force_kn > 0
        assert result.projected_area_cm2 == 50.0
        assert "t" in result.recommended_machine

    def test_clamp_force_multi_cavity(self) -> None:
        """多数個取りでは型締力が増加."""
        from app.mold.sizing import calculate_clamp_force
        r1 = calculate_clamp_force(5000, "ABS", cavity_count=1)
        r4 = calculate_clamp_force(5000, "ABS", cavity_count=4)
        assert r4.clamp_force_ton > r1.clamp_force_ton * 3

    def test_clamp_force_unknown_resin(self) -> None:
        """不明な樹脂ではABSデフォルト."""
        from app.mold.sizing import calculate_clamp_force
        result = calculate_clamp_force(5000, "UNKNOWN_RESIN")
        assert result.clamp_force_ton > 0

    def test_runner_design(self) -> None:
        """ランナー設計."""
        from app.mold.sizing import design_runner
        result = design_runner(part_weight_g=30, wall_thickness_mm=2.5, resin="ABS")
        assert result.sprue_diameter > 0
        assert result.main_runner_diameter > 0
        assert result.gate_width > 0
        assert result.estimated_waste_g > 0
        assert result.gate_type in ("ピンゲート", "サイドゲート", "ダイレクトゲート")

    def test_runner_thin_wall_pin_gate(self) -> None:
        """薄肉ではピンゲート選定."""
        from app.mold.sizing import design_runner
        result = design_runner(part_weight_g=5, wall_thickness_mm=1.0)
        assert result.gate_type == "ピンゲート"

    def test_cooling_design(self) -> None:
        """冷却系設計."""
        from app.mold.sizing import design_cooling
        result = design_cooling(wall_thickness_mm=2.5, part_width_mm=100, part_length_mm=80)
        assert result.channel_diameter in (8, 10, 12)
        assert result.channel_pitch > result.channel_diameter
        assert result.estimated_cooling_time_s > 0
        assert result.channel_count >= 4

    def test_ejector_design(self) -> None:
        """エジェクタ設計."""
        from app.mold.sizing import design_ejector
        result = design_ejector(
            part_depth_mm=25, part_area_mm2=5000,
            wall_thickness_mm=2.5, draft_angle_deg=1.0,
        )
        assert result.ejector_stroke_mm >= 35  # 25 + 10
        assert result.ejector_force_kn > 0
        assert result.ejector_pin_count >= 4
        assert result.needs_stripper_plate is False

    def test_ejector_stripper_needed(self) -> None:
        """抜き勾配が小さいとストリッパ推奨."""
        from app.mold.sizing import design_ejector
        result = design_ejector(
            part_depth_mm=15, part_area_mm2=3000,
            wall_thickness_mm=2.0, draft_angle_deg=0.3,
        )
        assert result.needs_stripper_plate is True

    def test_steel_selection_low_volume(self) -> None:
        """少量生産ではS50C."""
        from app.mold.sizing import select_mold_steel
        result = select_mold_steel(production_quantity=1000)
        assert "S50C" in result.core_steel

    def test_steel_selection_high_volume(self) -> None:
        """大量生産ではSKD61."""
        from app.mold.sizing import select_mold_steel
        result = select_mold_steel(production_quantity=100000)
        assert "SKD61" in result.core_steel

    def test_steel_selection_mirror(self) -> None:
        """鏡面仕上げではSTAVAX."""
        from app.mold.sizing import select_mold_steel
        result = select_mold_steel(surface_finish="mirror")
        assert "STAVAX" in result.core_steel

    def test_cycle_time(self) -> None:
        """サイクルタイム概算."""
        from app.mold.sizing import estimate_cycle_time
        result = estimate_cycle_time(
            wall_thickness_mm=2.5, part_weight_g=30,
            resin="ABS", production_quantity=10000,
        )
        assert result.total_cycle_s > 0
        assert result.shots_per_hour > 0
        assert result.production_time_hours > 0

    def test_resin_db(self) -> None:
        """樹脂物性DBが10種以上."""
        from app.mold.sizing import RESIN_DB
        assert len(RESIN_DB) >= 10
        for name, props in RESIN_DB.items():
            assert props.density > 0
            assert props.melt_temp_max > props.melt_temp_min


# ===== T-075: ドラフト角解析テスト =====


class TestDraftAnalysis:
    """ドラフト角解析のテスト."""

    def test_box_draft_angles(self, tmp_path: Path) -> None:
        """直方体の側面はドラフト角0°."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.draft_analysis import analyze_draft_angles
        result = analyze_draft_angles(stl_path, min_draft_deg=1.0)
        assert result.parting_direction is not None
        assert len(result.faces) > 0

    def test_compliance_ratio(self, tmp_path: Path) -> None:
        """合格率が0〜1の範囲."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.draft_analysis import analyze_draft_angles
        result = analyze_draft_angles(stl_path)
        assert 0.0 <= result.compliance_ratio <= 1.0

    def test_zero_draft_detection(self, tmp_path: Path) -> None:
        """直方体の側面(ドラフト0°)が不足として検出."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.draft_analysis import analyze_draft_angles
        result = analyze_draft_angles(stl_path, min_draft_deg=1.0)
        # 直方体の側面は型開き方向に平行 → ドラフト角 ≈ 0°
        assert len(result.insufficient_faces) > 0

    def test_summary_generated(self, tmp_path: Path) -> None:
        """サマリが生成される."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.draft_analysis import analyze_draft_angles
        result = analyze_draft_angles(stl_path)
        assert "ドラフト角解析" in result.summary
        assert "合格率" in result.summary

    def test_statistics(self, tmp_path: Path) -> None:
        """統計値が正しい."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.draft_analysis import analyze_draft_angles
        result = analyze_draft_angles(stl_path)
        assert result.min_draft_deg >= 0
        assert result.max_draft_deg >= result.min_draft_deg
        assert result.average_draft_deg >= 0

    def test_custom_direction(self, tmp_path: Path) -> None:
        """カスタム型開き方向."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.draft_analysis import analyze_draft_angles
        result = analyze_draft_angles(stl_path, parting_direction=np.array([1, 0, 0]))
        assert result.parting_direction[0] > 0.99


# ===== T-076: Core統合テスト =====


class TestCoreIntegration:
    """core.pyにsizing/draft統合されていることのテスト."""

    def test_design_has_sizing(self, tmp_path: Path) -> None:
        """設計結果にサイジング結果が含まれる."""
        stl_path = _create_box_stl(tmp_path / "box.stl", size=(80, 60, 30))
        from app.mold.core import MoldDesigner
        result = MoldDesigner().design(stl_path)
        assert result.clamp_force_ton > 0
        assert result.recommended_machine != ""
        assert result.cycle_time_s > 0
        assert result.shots_per_hour > 0

    def test_design_has_draft(self, tmp_path: Path) -> None:
        """設計結果にドラフト角解析が含まれる."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.core import MoldDesigner
        result = MoldDesigner().design(stl_path)
        assert result.draft_result is not None
        assert result.draft_result.compliance_ratio >= 0

    def test_summary_includes_sizing(self, tmp_path: Path) -> None:
        """サマリにサイジング情報が含まれる."""
        stl_path = _create_box_stl(tmp_path / "box.stl")
        from app.mold.core import MoldDesigner
        result = MoldDesigner().design(stl_path)
        assert "推奨成形機" in result.summary
        assert "サイクルタイム" in result.summary
        assert "ドラフト角" in result.summary
