"""
メッシュ修復モジュール.

生成された3Dメッシュの品質改善:
- 穴の充填
- 自己交差の除去
- 面の方向統一
- メッシュの簡略化
- ウォータータイト化

Implements: F-061 (メッシュ修復)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import CAD3DError

logger = logging.getLogger(__name__)


@dataclass
class MeshRepairConfig:
    """メッシュ修復の設定."""
    fill_holes: bool = True
    remove_self_intersections: bool = True
    fix_normals: bool = True
    simplify: bool = False
    target_faces: int = 50000
    make_watertight: bool = False
    smooth: bool = False
    smooth_iterations: int = 3


@dataclass
class MeshStats:
    """メッシュの統計情報."""
    vertices: int = 0
    faces: int = 0
    edges: int = 0
    is_watertight: bool = False
    has_degenerate_faces: int = 0
    volume: float = 0.0
    surface_area: float = 0.0


def repair_mesh(
    input_path: Path,
    output_path: Path,
    config: MeshRepairConfig | None = None,
) -> tuple[Path, MeshStats]:
    """
    3Dメッシュを修復.

    Args:
        input_path: 入力メッシュファイルパス.
        output_path: 出力メッシュファイルパス.
        config: 修復設定.

    Returns:
        (修復済みメッシュパス, メッシュ統計情報).
    """
    if config is None:
        config = MeshRepairConfig()

    try:
        import trimesh
    except ImportError:
        raise CAD3DError("trimesh がインストールされていません", detail="pip install trimesh")

    # メッシュをロード
    mesh = trimesh.load(str(input_path))
    if not isinstance(mesh, trimesh.Trimesh):
        if isinstance(mesh, trimesh.Scene):
            # シーンの場合は最大メッシュを取得
            geometries = list(mesh.geometry.values())
            if not geometries:
                raise CAD3DError("メッシュが見つかりません")
            mesh = max(geometries, key=lambda g: len(g.faces) if hasattr(g, 'faces') else 0)
        else:
            raise CAD3DError(f"サポートされていないメッシュ形式: {type(mesh)}")

    logger.info(
        "Input mesh: %d vertices, %d faces, watertight=%s",
        len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
    )

    # 修復ステップ
    if config.fix_normals:
        mesh.fix_normals()
        logger.info("Fixed normals")

    if config.fill_holes:
        mesh.fill_holes()
        logger.info("Filled holes")

    if config.remove_self_intersections:
        # 退化面の除去
        mask = mesh.face_adjacency_angles < 0.001
        if mask.any():
            mesh.update_faces(~trimesh.grouping.boolean_rows(
                mesh.face_adjacency, mask, len(mesh.faces)
            ))
            logger.info("Removed degenerate faces")

    if config.simplify and config.target_faces < len(mesh.faces):
        try:
            mesh = mesh.simplify_quadric_decimation(config.target_faces)
            logger.info("Simplified to %d faces", len(mesh.faces))
        except Exception:
            logger.warning("Simplification failed, skipping")

    if config.make_watertight:
        try:
            import pymeshfix
            tin = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
            tin.repair()
            mesh = trimesh.Trimesh(vertices=tin.v, faces=tin.f)
            logger.info("Made watertight with PyMeshFix")
        except ImportError:
            logger.warning("pymeshfix がインストールされていません。ウォータータイト化をスキップ。")

    if config.smooth:
        mesh = trimesh.smoothing.filter_laplacian(
            mesh, iterations=config.smooth_iterations
        )
        logger.info("Applied Laplacian smoothing (%d iterations)", config.smooth_iterations)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    # 統計情報
    stats = MeshStats(
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        edges=len(mesh.edges) if hasattr(mesh, 'edges') else 0,
        is_watertight=mesh.is_watertight,
        has_degenerate_faces=len(mesh.degenerate_faces) if hasattr(mesh, 'degenerate_faces') else 0,
        volume=float(mesh.volume) if mesh.is_watertight else 0.0,
        surface_area=float(mesh.area),
    )

    logger.info(
        "Output mesh: %d vertices, %d faces, watertight=%s",
        stats.vertices, stats.faces, stats.is_watertight,
    )

    return output_path, stats


def get_mesh_info(mesh_path: Path) -> MeshStats:
    """メッシュの統計情報を取得."""
    try:
        import trimesh
        mesh = trimesh.load(str(mesh_path))
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            if geometries:
                mesh = max(geometries, key=lambda g: len(g.faces) if hasattr(g, 'faces') else 0)
            else:
                return MeshStats()

        return MeshStats(
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            edges=len(mesh.edges) if hasattr(mesh, 'edges') else 0,
            is_watertight=mesh.is_watertight,
            volume=float(mesh.volume) if mesh.is_watertight else 0.0,
            surface_area=float(mesh.area),
        )
    except Exception:
        return MeshStats()
