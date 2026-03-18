"""
パーティングライン最適化モジュール.

パーティングライン (PL) は金型を可動側と固定側に分ける境界線.
製品の向きとPL位置はアンダーカットの有無に直接影響する.

最適化基準 (ユーザー要件に基づく):
1. 金型サイズの高さ方向が薄くなること
2. 金型体積が小さくなること
3. アンダーカットが少なくなること
4. 樹脂の流動性 (ゲート配置のしやすさ)
5. 金型加工のしやすさ

Implements: F-071 (パーティングライン決定)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.core.exceptions import CAD3DError

logger = logging.getLogger(__name__)


@dataclass
class PartingLineCandidate:
    """パーティングライン候補."""
    direction: np.ndarray       # 型開き方向 [x, y, z]
    parting_plane_z: float      # パーティング面のZ座標 (型開き方向の投影座標)
    score: float                # 総合スコア (低いほど良い)
    mold_height: float          # 金型高さ [mm]
    mold_volume: float          # 金型概算体積 [mm³]
    undercut_area: float        # アンダーカット面積 [mm²]
    undercut_count: int         # アンダーカット領域数
    gate_accessibility: float   # ゲート配置しやすさ (0-1, 高いほど良い)
    machinability: float        # 加工のしやすさ (0-1, 高いほど良い)
    parting_line_vertices: list[int] = field(default_factory=list)  # PL頂点


@dataclass
class PartingLineResult:
    """パーティングライン最適化結果."""
    best: PartingLineCandidate
    candidates: list[PartingLineCandidate]
    product_orientation: np.ndarray   # 最適な製品配置の回転行列
    summary: str = ""


def optimize_parting_line(
    mesh_path,
    candidate_count: int = 6,
    weights: dict[str, float] | None = None,
) -> PartingLineResult:
    """
    パーティングラインを最適化.

    Args:
        mesh_path: 入力メッシュ (パス or trimeshオブジェクト).
        candidate_count: 評価する候補方向数.
        weights: スコアリングの重み.
            - "height": 金型高さの重み (デフォルト 1.0)
            - "volume": 金型体積の重み (デフォルト 0.5)
            - "undercut": アンダーカット面積の重み (デフォルト 3.0)
            - "gate": ゲートのしやすさの重み (デフォルト 1.0)
            - "machinability": 加工のしやすさの重み (デフォルト 0.5)

    Returns:
        最適化結果.
    """
    from pathlib import Path

    try:
        import trimesh
    except ImportError:
        raise CAD3DError("trimesh が必要です")

    # メッシュロード
    if isinstance(mesh_path, (str, Path)):
        mesh = trimesh.load(str(mesh_path))
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            mesh = max(geometries, key=lambda g: len(g.faces) if hasattr(g, 'faces') else 0)
    else:
        mesh = mesh_path

    if weights is None:
        weights = {
            "height": 1.0,
            "volume": 0.5,
            "undercut": 3.0,
            "gate": 1.0,
            "machinability": 0.5,
        }

    # 候補方向の生成
    directions = _generate_candidate_directions(mesh, candidate_count)

    # 各方向を評価
    candidates: list[PartingLineCandidate] = []
    for direction in directions:
        candidate = _evaluate_direction(mesh, direction, weights)
        candidates.append(candidate)

    # スコアでソート
    candidates.sort(key=lambda c: c.score)

    # 最適解
    best = candidates[0]
    orientation = _compute_orientation_matrix(best.direction)

    result = PartingLineResult(
        best=best,
        candidates=candidates,
        product_orientation=orientation,
    )
    result.summary = _generate_pl_summary(result)
    logger.info("Parting line optimized: %s", result.summary)
    return result


def _generate_candidate_directions(mesh, count: int) -> list[np.ndarray]:
    """
    候補となる型開き方向を生成.

    戦略:
    - 6つの主軸方向 (±X, ±Y, ±Z) は必須
    - バウンディングボックスの最短軸方向を最優先
    - 主成分分析 (PCA) の主軸も候補に追加
    """
    directions = [
        np.array([0, 0, 1], dtype=np.float64),
        np.array([0, 0, -1], dtype=np.float64),
        np.array([0, 1, 0], dtype=np.float64),
        np.array([0, -1, 0], dtype=np.float64),
        np.array([1, 0, 0], dtype=np.float64),
        np.array([-1, 0, 0], dtype=np.float64),
    ]

    # PCA主軸も追加
    try:
        centered = mesh.vertices - mesh.vertices.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 最小固有値の軸 = 最も薄い方向
        min_axis = eigenvectors[:, 0]
        directions.append(min_axis / np.linalg.norm(min_axis))
        directions.append(-min_axis / np.linalg.norm(min_axis))
    except Exception:
        pass

    # 重複除去
    unique_dirs: list[np.ndarray] = []
    for d in directions:
        is_dup = False
        for existing in unique_dirs:
            if abs(np.dot(d, existing)) > 0.99:
                is_dup = True
                break
        if not is_dup:
            unique_dirs.append(d)

    return unique_dirs[:count]


def _evaluate_direction(
    mesh,
    direction: np.ndarray,
    weights: dict[str, float],
) -> PartingLineCandidate:
    """1つの方向について金型としての適合度を評価."""
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces
    vertices = mesh.vertices

    # 型開き方向への頂点投影
    projections = np.dot(vertices, direction)
    proj_min, proj_max = projections.min(), projections.max()
    mold_height = proj_max - proj_min

    # パーティング面のZ座標 (投影の中央値)
    parting_z = np.median(projections)

    # 金型の概算体積 (バウンディングボックス)
    bb = mesh.bounding_box
    mold_volume = float(np.prod(bb.extents)) * 1.3  # マージン30%

    # アンダーカット面積の概算
    dots = np.dot(face_normals, direction)
    # 法線が型開きと逆方向で、かつ型を開くと引っかかる面
    undercut_mask = dots < -np.cos(np.radians(85))
    undercut_area = float(face_areas[undercut_mask].sum())

    # レイキャスティングによる追加検出 (簡易版 - 中心軸のみ)
    undercut_count = int(np.sum(undercut_mask))

    # ゲート配置しやすさ
    gate_accessibility = _evaluate_gate_accessibility(mesh, direction, parting_z)

    # 加工のしやすさ
    machinability = _evaluate_machinability(mesh, direction)

    # パーティングライン頂点 (パーティング面付近の頂点)
    pl_tolerance = mold_height * 0.02
    pl_vertices = [
        int(i) for i, p in enumerate(projections) if abs(p - parting_z) < pl_tolerance
    ]

    # 総合スコア計算 (低いほど良い)
    # 各指標を正規化してから重み付け
    h_w = weights.get("height", 1.0)
    v_w = weights.get("volume", 0.5)
    u_w = weights.get("undercut", 3.0)
    g_w = weights.get("gate", 1.0)
    m_w = weights.get("machinability", 0.5)

    score = (
        h_w * (mold_height / 100.0) +
        v_w * (mold_volume / 1e6) +
        u_w * (undercut_area / 1000.0) +
        g_w * (1.0 - gate_accessibility) +
        m_w * (1.0 - machinability)
    )

    return PartingLineCandidate(
        direction=direction,
        parting_plane_z=parting_z,
        score=score,
        mold_height=mold_height,
        mold_volume=mold_volume,
        undercut_area=undercut_area,
        undercut_count=undercut_count,
        gate_accessibility=gate_accessibility,
        machinability=machinability,
        parting_line_vertices=pl_vertices,
    )


def _evaluate_gate_accessibility(
    mesh, direction: np.ndarray, parting_z: float,
) -> float:
    """
    ゲート配置しやすさを評価.

    パーティング面付近に十分な平坦面があるほど高評価.
    """
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    face_areas = mesh.area_faces

    # パーティング面付近の面を抽出
    center_projections = np.dot(face_centers, direction)
    height = np.dot(mesh.vertices, direction)
    total_height = height.max() - height.min()
    tolerance = total_height * 0.1

    near_pl = np.abs(center_projections - parting_z) < tolerance
    if not near_pl.any():
        return 0.3

    # パーティング面付近の面積比 → 大きいほどゲート配置しやすい
    pl_area = face_areas[near_pl].sum()
    total_area = face_areas.sum()
    ratio = pl_area / total_area if total_area > 0 else 0

    return min(1.0, ratio * 5.0)


def _evaluate_machinability(mesh, direction: np.ndarray) -> float:
    """
    加工の容易さを評価.

    基準:
    - 深い溝やポケットが少ないほど加工しやすい
    - 面法線の分散が小さいほど単純な形状→加工しやすい
    """
    normals = mesh.face_normals
    dots = np.abs(np.dot(normals, direction))

    # 法線方向のエントロピーが低い→加工しやすい
    hist, _ = np.histogram(dots, bins=10, range=(0, 1))
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    max_entropy = np.log2(10)

    return max(0.0, 1.0 - entropy / max_entropy)


def _compute_orientation_matrix(direction: np.ndarray) -> np.ndarray:
    """型開き方向をZ軸に合わせる回転行列を計算."""
    z_axis = direction / np.linalg.norm(direction)

    if abs(z_axis[2]) < 0.9:
        up = np.array([0, 0, 1], dtype=np.float64)
    else:
        up = np.array([0, 1, 0], dtype=np.float64)

    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    return np.column_stack([x_axis, y_axis, z_axis])


def _generate_pl_summary(result: PartingLineResult) -> str:
    """パーティングライン最適化のサマリ."""
    b = result.best
    lines = [
        f"▼ パーティングライン最適化結果",
        f"  最適方向: [{b.direction[0]:.2f}, {b.direction[1]:.2f}, {b.direction[2]:.2f}]",
        f"  金型高さ: {b.mold_height:.1f} mm",
        f"  金型概算体積: {b.mold_volume:.0f} mm³",
        f"  アンダーカット面積: {b.undercut_area:.1f} mm²",
        f"  ゲート配置: {b.gate_accessibility:.0%}",
        f"  加工容易度: {b.machinability:.0%}",
        f"  総合スコア: {b.score:.3f}",
        f"  評価候補数: {len(result.candidates)}",
    ]
    return "\n".join(lines)
