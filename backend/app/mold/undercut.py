"""
アンダーカット自動検出モジュール.

樹脂成形品の3Dメッシュからアンダーカット形状を自動検出する.

手法:
1. 法線ベクトル分析: 各面の法線とパーティング方向の内積 → 負 = アンダーカット候補
2. レイキャスティング: 型開き方向にレイを発射し、自己遮蔽 = 離型不可
3. 可視性分析: パーティング方向からの面アクセシビリティを分類

参考文献:
- "Automatic Generation of Core and Cavity for 3D CAD Model"
  手法: "scanning ray approach to detect through holes and undercuts"
  訳: レイ走査法で貫通穴とアンダーカットを検出
  (ARPN Journal of Engineering and Applied Sciences)

- "Undercut-Free Parting Direction Determination"
  手法: "accessibility of each surface ... rays are cast along the
  chosen parting direction to analyze the accessibility"
  訳: 各面のアクセシビリティを、選択したパーティング方向に沿って
  レイを発射して分析する
  (Computer-Aided Design and Applications)

Implements: F-070 (アンダーカット検出)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from app.core.exceptions import CAD3DError

logger = logging.getLogger(__name__)


class FaceClassification(str, Enum):
    """面の分類."""
    CORE = "core"           # 可動側 (型開き方向と同じ側)
    CAVITY = "cavity"       # 固定側
    UNDERCUT = "undercut"   # アンダーカット (型開き方向に対して遮蔽)
    PARTING = "parting"     # パーティング面 (型開き方向に垂直)
    THROUGH_HOLE = "through_hole"  # 貫通穴


@dataclass
class UndercutRegion:
    """検出されたアンダーカット領域."""
    face_indices: list[int]
    area: float                     # アンダーカット面積 [mm²]
    centroid: np.ndarray            # 重心位置
    depth: float                    # アンダーカット深さ [mm]
    direction: np.ndarray           # アンダーカット方向 (型開きと交差する方向)
    recommended_mechanism: str      # "slide_core", "angled_lifter", "collapsible_core"
    severity: str                   # "minor", "moderate", "major"


@dataclass
class UndercutAnalysisResult:
    """アンダーカット解析結果."""
    parting_direction: np.ndarray   # 型開き方向
    face_classifications: np.ndarray  # 各面の分類
    undercut_regions: list[UndercutRegion] = field(default_factory=list)
    total_undercut_area: float = 0.0
    undercut_count: int = 0
    has_undercut: bool = False
    core_faces: list[int] = field(default_factory=list)
    cavity_faces: list[int] = field(default_factory=list)
    summary: str = ""


def detect_undercuts(
    mesh_path: Path,
    parting_direction: np.ndarray | None = None,
    ray_density: int = 50,
    angle_threshold: float = 5.0,
) -> UndercutAnalysisResult:
    """
    3Dメッシュからアンダーカットを検出.

    Args:
        mesh_path: 入力メッシュファイルパス (STL/OBJ/GLB等).
        parting_direction: 型開き方向ベクトル (Noneの場合は自動決定).
        ray_density: レイキャスティング密度 (1軸あたりのレイ本数).
        angle_threshold: パーティング面判定の角度閾値 [度].

    Returns:
        アンダーカット解析結果.
    """
    try:
        import trimesh
    except ImportError:
        raise CAD3DError("trimesh が必要です", detail="pip install trimesh")

    # メッシュロード
    mesh = _load_mesh(mesh_path)

    # 型開き方向が未指定の場合は自動決定
    if parting_direction is None:
        parting_direction = _auto_determine_parting_direction(mesh)
    else:
        parting_direction = np.array(parting_direction, dtype=np.float64)
        parting_direction = parting_direction / np.linalg.norm(parting_direction)

    logger.info("Parting direction: %s", parting_direction)

    # Step 1: 法線ベクトル分析による面分類
    classifications = _classify_faces_by_normal(
        mesh, parting_direction, angle_threshold
    )

    # Step 2: レイキャスティングによる遮蔽検出
    occluded_faces = _ray_cast_occlusion(
        mesh, parting_direction, ray_density
    )

    # Step 3: 遮蔽された面をアンダーカットに更新
    for fi in occluded_faces:
        if classifications[fi] not in (FaceClassification.PARTING,):
            classifications[fi] = FaceClassification.UNDERCUT

    # Step 4: アンダーカット領域のクラスタリング
    undercut_indices = [
        i for i, c in enumerate(classifications) if c == FaceClassification.UNDERCUT
    ]
    undercut_regions = _cluster_undercut_regions(
        mesh, undercut_indices, parting_direction
    )

    # 結果集計
    core_faces = [i for i, c in enumerate(classifications) if c == FaceClassification.CORE]
    cavity_faces = [i for i, c in enumerate(classifications) if c == FaceClassification.CAVITY]
    total_area = sum(r.area for r in undercut_regions)

    result = UndercutAnalysisResult(
        parting_direction=parting_direction,
        face_classifications=np.array([c.value for c in classifications]),
        undercut_regions=undercut_regions,
        total_undercut_area=total_area,
        undercut_count=len(undercut_regions),
        has_undercut=len(undercut_regions) > 0,
        core_faces=core_faces,
        cavity_faces=cavity_faces,
    )

    # サマリ生成
    result.summary = _generate_summary(result, mesh)
    logger.info("Undercut analysis: %s", result.summary)
    return result


def _load_mesh(mesh_path: Path):
    """メッシュをロード (Scene→単一Trimesh変換含む)."""
    import trimesh

    mesh = trimesh.load(str(mesh_path))
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise CAD3DError("メッシュが見つかりません")
        mesh = max(geometries, key=lambda g: len(g.faces) if hasattr(g, 'faces') else 0)

    if not isinstance(mesh, trimesh.Trimesh):
        raise CAD3DError(f"サポートされていない形式: {type(mesh)}")

    return mesh


def _auto_determine_parting_direction(mesh) -> np.ndarray:
    """
    製品形状から最適な型開き方向を自動決定.

    戦略:
    1. バウンディングボックスの最短軸 → 金型高さが最小になる方向
    2. 各主軸候補についてアンダーカット面積を概算
    3. アンダーカット最小の方向を選択

    製品のXY軸平行方向を基本とする設計思想に基づき、
    Z軸方向を優先候補とする.
    """
    extents = mesh.bounding_box.extents
    candidates = [
        np.array([0, 0, 1], dtype=np.float64),   # Z+ (最も一般的)
        np.array([0, 0, -1], dtype=np.float64),  # Z-
        np.array([0, 1, 0], dtype=np.float64),   # Y+
        np.array([1, 0, 0], dtype=np.float64),   # X+
    ]

    # バウンディングボックス最短軸を最優先候補に
    min_axis = int(np.argmin(extents))
    primary = np.zeros(3)
    primary[min_axis] = 1.0
    candidates.insert(0, primary)

    best_direction = candidates[0]
    best_score = float("inf")

    face_normals = mesh.face_normals
    face_areas = mesh.area_faces

    for direction in candidates:
        # スコア = アンダーカット面積 + 金型高さペナルティ
        dots = np.dot(face_normals, direction)

        # アンダーカット候補面積 (法線が型開き方向と逆向きの面)
        undercut_mask = dots < -np.cos(np.radians(85))
        undercut_area = face_areas[undercut_mask].sum()

        # 金型高さ (この方向への製品投影高さ)
        vertices_proj = np.dot(mesh.vertices, direction)
        mold_height = vertices_proj.max() - vertices_proj.min()

        # 総合スコア (アンダーカット面積が重要、高さは補助)
        score = undercut_area * 100.0 + mold_height
        if score < best_score:
            best_score = score
            best_direction = direction

    logger.info("Auto-determined parting direction: %s (score=%.2f)", best_direction, best_score)
    return best_direction


def _classify_faces_by_normal(
    mesh,
    parting_dir: np.ndarray,
    angle_threshold: float,
) -> list[FaceClassification]:
    """
    法線ベクトルとパーティング方向の内積で面を分類.

    内積 > 0: Core面 (可動側)
    内積 < 0: Cavity面 (固定側)
    |内積| ≈ 0: Parting面 (パーティング面)
    """
    face_normals = mesh.face_normals
    dots = np.dot(face_normals, parting_dir)
    threshold_cos = np.cos(np.radians(90 - angle_threshold))

    classifications = []
    for dot_val in dots:
        if abs(dot_val) < threshold_cos:
            classifications.append(FaceClassification.PARTING)
        elif dot_val > 0:
            classifications.append(FaceClassification.CORE)
        else:
            classifications.append(FaceClassification.CAVITY)

    return classifications


def _ray_cast_occlusion(
    mesh,
    parting_dir: np.ndarray,
    ray_density: int,
) -> set[int]:
    """
    レイキャスティングで自己遮蔽面 (アンダーカット) を検出.

    論文手法:
    "scanning ray approach to detect through holes and undercuts"
    (ARPN Journal)

    実装:
    1. メッシュのバウンディングボックス上面からレイを下向きに発射
    2. ヒットした面のうち、法線が型開き方向を見ていない面を検出
    3. 複数回ヒットする領域 = 凹み/穴/アンダーカット
    """
    import trimesh

    bounds = mesh.bounds
    margin = (bounds[1] - bounds[0]).max() * 0.1

    # レイの発射面を生成 (パーティング方向と垂直な平面上のグリッド)
    ray_origins, ray_directions = _create_ray_grid(
        bounds, parting_dir, ray_density, margin
    )

    # レイキャスティング実行
    occluded_faces: set[int] = set()

    if len(ray_origins) == 0:
        return occluded_faces

    # trimesh の ray-mesh 交差判定
    # pyembree (高速) → 標準 ray (フォールバック) の順で試行
    try:
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector
            intersector = RayMeshIntersector(mesh)
            locations, index_ray, index_tri = intersector.intersects_location(
                ray_origins, ray_directions, multiple_hits=True
            )
        except (ImportError, ModuleNotFoundError, Exception):
            # pyembree/embreex 未インストール → 標準レイエンジン
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins, ray_directions, multiple_hits=True
            )
    except Exception:
        logger.warning("レイキャスティングに失敗、法線分析のみで判定")
        return occluded_faces

    # 各レイについて、複数ヒットがある場合を分析
    ray_hits: dict[int, list[tuple[float, int]]] = {}
    for loc, ri, ti in zip(locations, index_ray, index_tri):
        depth = np.dot(loc - ray_origins[ri], ray_directions[ri])
        ray_hits.setdefault(ri, []).append((depth, ti))

    for ray_id, hits in ray_hits.items():
        if len(hits) <= 1:
            continue

        # ヒットを深さ順にソート
        hits.sort(key=lambda x: x[0])

        # 最初のヒット以降で、法線がレイ方向と同方向の面 = アンダーカット
        for i, (depth, face_id) in enumerate(hits):
            if i == 0:
                continue
            face_normal = mesh.face_normals[face_id]
            dot = np.dot(face_normal, parting_dir)
            # 面の法線が型開き方向を向いている(内面) = 引っかかる
            if dot > 0.1:
                occluded_faces.add(face_id)

    logger.info("Ray casting: %d occluded faces detected", len(occluded_faces))
    return occluded_faces


def _create_ray_grid(
    bounds: np.ndarray,
    parting_dir: np.ndarray,
    density: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """パーティング方向に垂直な平面上にレイのグリッドを生成."""
    # パーティング方向に垂直な2つの基底ベクトル
    if abs(parting_dir[2]) < 0.9:
        up = np.array([0, 0, 1], dtype=np.float64)
    else:
        up = np.array([1, 0, 0], dtype=np.float64)

    u = np.cross(parting_dir, up)
    u = u / np.linalg.norm(u)
    v = np.cross(parting_dir, u)
    v = v / np.linalg.norm(v)

    # バウンディングボックスの中心
    center = (bounds[0] + bounds[1]) / 2.0
    extent = bounds[1] - bounds[0]
    max_extent = np.linalg.norm(extent)

    # レイの発射位置 (パーティング方向の反対側から)
    start_offset = center - parting_dir * (max_extent + margin)

    origins = []
    directions = []

    half_range = max_extent / 2 + margin
    step = 2 * half_range / density

    for i in range(density):
        for j in range(density):
            offset_u = -half_range + i * step
            offset_v = -half_range + j * step
            origin = start_offset + u * offset_u + v * offset_v
            origins.append(origin)
            directions.append(parting_dir.copy())

    return np.array(origins), np.array(directions)


def _cluster_undercut_regions(
    mesh,
    undercut_indices: list[int],
    parting_dir: np.ndarray,
) -> list[UndercutRegion]:
    """
    アンダーカット面をグラフベースでクラスタリング.

    隣接するアンダーカット面を1つの領域としてグループ化.
    """
    if not undercut_indices:
        return []

    undercut_set = set(undercut_indices)

    # 面隣接グラフを構築
    adjacency: dict[int, set[int]] = {}
    face_adjacency = mesh.face_adjacency
    for edge in face_adjacency:
        f0, f1 = int(edge[0]), int(edge[1])
        if f0 in undercut_set and f1 in undercut_set:
            adjacency.setdefault(f0, set()).add(f1)
            adjacency.setdefault(f1, set()).add(f0)

    # 連結成分の検出 (BFS)
    visited: set[int] = set()
    clusters: list[list[int]] = []

    for fi in undercut_indices:
        if fi in visited:
            continue
        cluster: list[int] = []
        queue = [fi]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        if cluster:
            clusters.append(cluster)

    # 各クラスタをUndercutRegionに変換
    regions: list[UndercutRegion] = []
    face_areas = mesh.area_faces
    face_normals = mesh.face_normals

    for cluster in clusters:
        cluster_arr = np.array(cluster)
        area = float(face_areas[cluster_arr].sum())

        # 重心計算
        centroids = mesh.triangles_center[cluster_arr]
        centroid = centroids.mean(axis=0)

        # アンダーカット深さ (パーティング方向への投影距離)
        verts_in_cluster = set()
        for fi in cluster:
            for vi in mesh.faces[fi]:
                verts_in_cluster.add(vi)
        if verts_in_cluster:
            projections = [np.dot(mesh.vertices[vi], parting_dir) for vi in verts_in_cluster]
            depth = max(projections) - min(projections)
        else:
            depth = 0.0

        # アンダーカット方向 (平均法線のパーティング方向垂直成分)
        avg_normal = face_normals[cluster_arr].mean(axis=0)
        avg_normal = avg_normal - np.dot(avg_normal, parting_dir) * parting_dir
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-6:
            avg_normal = avg_normal / norm
        else:
            avg_normal = np.array([1, 0, 0], dtype=np.float64)

        # 推奨メカニズムの判定
        mechanism = _recommend_mechanism(area, depth)

        # 深刻度
        if area > 500 or depth > 20:
            severity = "major"
        elif area > 100 or depth > 5:
            severity = "moderate"
        else:
            severity = "minor"

        regions.append(UndercutRegion(
            face_indices=cluster,
            area=area,
            centroid=centroid,
            depth=depth,
            direction=avg_normal,
            recommended_mechanism=mechanism,
            severity=severity,
        ))

    # 面積降順でソート
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def _recommend_mechanism(area: float, depth: float) -> str:
    """
    アンダーカットに対する推奨メカニズムを判定.

    ルール:
    - 浅い (depth < 3mm) + 小面積: 傾斜コア (angled lifter)
    - 深い or 大面積: スライドコア (slide core)
    - 円形/環状: 折りたたみコア (collapsible core)
    """
    if depth < 3.0 and area < 50.0:
        return "angled_lifter"
    elif depth > 15.0 or area > 300.0:
        return "slide_core"
    else:
        return "slide_core"


def _generate_summary(result: UndercutAnalysisResult, mesh) -> str:
    """解析サマリを生成."""
    lines = []
    lines.append(f"▼ アンダーカット解析結果")
    lines.append(f"  パーティング方向: [{result.parting_direction[0]:.2f}, "
                 f"{result.parting_direction[1]:.2f}, {result.parting_direction[2]:.2f}]")
    lines.append(f"  Core 面数: {len(result.core_faces)}")
    lines.append(f"  Cavity 面数: {len(result.cavity_faces)}")
    lines.append(f"  アンダーカット: {'あり' if result.has_undercut else 'なし'}")

    if result.undercut_regions:
        lines.append(f"  アンダーカット領域数: {result.undercut_count}")
        lines.append(f"  総アンダーカット面積: {result.total_undercut_area:.1f} mm²")
        for i, region in enumerate(result.undercut_regions):
            lines.append(f"  [{i+1}] 面積={region.area:.1f}mm², "
                        f"深さ={region.depth:.1f}mm, "
                        f"深刻度={region.severity}, "
                        f"推奨={region.recommended_mechanism}")

    return "\n".join(lines)
