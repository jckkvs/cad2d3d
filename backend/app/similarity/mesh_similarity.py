"""
3Dメッシュ形状類似度比較モジュール.

メッシュ間の形状類似度をスコアリングし、過去の見積条件を参照しやすくする.

手法:
1. D2 Shape Distribution — メッシュ上のランダム2点間距離のヒストグラム
2. Moment Invariants — 慣性モーメントベースの形状記述子
3. バウンディングボックスアスペクト比

参考文献:
"Shape Distributions" by Osada et al. (2002)
  原文: "A shape distribution is a probability distribution sampled
  from a shape function measuring some property of a 3D model."
  訳: 形状分布は3Dモデルの特性を測定する形状関数からサンプリングした
  確率分布である.

Implements: F-080 (3Dメッシュ類似度)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.exceptions import CAD3DError

logger = logging.getLogger(__name__)


@dataclass
class MeshDescriptor:
    """メッシュの形状記述子."""
    d2_histogram: np.ndarray   # D2分布ヒストグラム (64bins)
    volume: float
    surface_area: float
    aspect_ratios: np.ndarray  # [w/h, w/d, h/d]
    compactness: float         # 球形度
    vertex_count: int
    face_count: int
    source_path: str = ""


@dataclass
class SimilarityResult:
    """類似度比較の結果."""
    score: float               # 総合類似度スコア (0-1, 高いほど類似)
    d2_similarity: float       # D2分布の類似度
    aspect_similarity: float   # アスペクト比の類似度
    volume_similarity: float   # 体積の類似度
    compactness_similarity: float


def compute_descriptor(mesh_path: Path, sample_count: int = 10000) -> MeshDescriptor:
    """
    メッシュの形状記述子を計算.

    Args:
        mesh_path: メッシュファイルパス.
        sample_count: D2分布のサンプル数.

    Returns:
        形状記述子.
    """
    try:
        import trimesh
    except ImportError:
        raise CAD3DError("trimesh が必要です")

    mesh = trimesh.load(str(mesh_path))
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        mesh = max(geometries, key=lambda g: len(g.faces) if hasattr(g, 'faces') else 0)

    # D2 Shape Distribution
    d2_hist = _compute_d2_distribution(mesh, sample_count)

    # 基本属性
    bb = mesh.bounding_box
    extents = sorted(bb.extents, reverse=True)
    aspect_ratios = np.array([
        extents[0] / extents[1] if extents[1] > 0 else 1.0,
        extents[0] / extents[2] if extents[2] > 0 else 1.0,
        extents[1] / extents[2] if extents[2] > 0 else 1.0,
    ])

    volume = float(mesh.volume) if mesh.is_watertight else float(np.prod(bb.extents) * 0.5)
    area = float(mesh.area)

    # 球形度 (Compactness)
    if area > 0:
        compactness = (36 * np.pi * volume**2) ** (1/3) / area
    else:
        compactness = 0.0

    return MeshDescriptor(
        d2_histogram=d2_hist,
        volume=volume,
        surface_area=area,
        aspect_ratios=aspect_ratios,
        compactness=compactness,
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        source_path=str(mesh_path),
    )


def compare_meshes(
    desc_a: MeshDescriptor,
    desc_b: MeshDescriptor,
    weights: dict[str, float] | None = None,
) -> SimilarityResult:
    """
    2つのメッシュの形状類似度を比較.

    Args:
        desc_a: メッシュAの記述子.
        desc_b: メッシュBの記述子.
        weights: 各指標の重み.

    Returns:
        類似度結果.
    """
    if weights is None:
        weights = {"d2": 0.4, "aspect": 0.2, "volume": 0.2, "compactness": 0.2}

    # D2ヒストグラムの類似度 (コサイン類似度)
    d2_sim = _cosine_similarity(desc_a.d2_histogram, desc_b.d2_histogram)

    # アスペクト比の類似度
    aspect_sim = 1.0 - np.mean(
        np.abs(desc_a.aspect_ratios - desc_b.aspect_ratios) /
        np.maximum(desc_a.aspect_ratios, desc_b.aspect_ratios)
    )
    aspect_sim = max(0.0, aspect_sim)

    # 体積の類似度
    v_max = max(desc_a.volume, desc_b.volume)
    volume_sim = min(desc_a.volume, desc_b.volume) / v_max if v_max > 0 else 1.0

    # 球形度の類似度
    c_max = max(desc_a.compactness, desc_b.compactness)
    compact_sim = min(desc_a.compactness, desc_b.compactness) / c_max if c_max > 0 else 1.0

    # 加重平均
    total_score = (
        weights.get("d2", 0.4) * d2_sim +
        weights.get("aspect", 0.2) * aspect_sim +
        weights.get("volume", 0.2) * volume_sim +
        weights.get("compactness", 0.2) * compact_sim
    )

    return SimilarityResult(
        score=float(np.clip(total_score, 0.0, 1.0)),
        d2_similarity=float(d2_sim),
        aspect_similarity=float(aspect_sim),
        volume_similarity=float(volume_sim),
        compactness_similarity=float(compact_sim),
    )


def compare_mesh_files(path_a: Path, path_b: Path) -> SimilarityResult:
    """ファイルパスから直接比較."""
    desc_a = compute_descriptor(path_a)
    desc_b = compute_descriptor(path_b)
    return compare_meshes(desc_a, desc_b)


def _compute_d2_distribution(mesh, sample_count: int, bins: int = 64) -> np.ndarray:
    """
    D2 Shape Distribution を計算.

    Osada et al. (2002):
    "D2 is the distance between two random points on the surface."
    訳: D2は表面上の2つのランダム点間距離である.
    """
    # メッシュ表面上のランダムサンプリング
    try:
        points, _ = trimesh_sample_surface(mesh, sample_count * 2)
    except Exception:
        # フォールバック: 頂点からランダムサンプリング
        indices = np.random.choice(len(mesh.vertices), size=min(sample_count * 2, len(mesh.vertices)), replace=True)
        points = mesh.vertices[indices]

    # ペア間距離の計算
    n = len(points) // 2
    if n == 0:
        return np.zeros(bins)

    p1 = points[:n]
    p2 = points[n:2*n]
    distances = np.linalg.norm(p1 - p2, axis=1)

    # 正規化 (バウンディングボックス対角長で正規化)
    diag = np.linalg.norm(mesh.bounding_box.extents)
    if diag > 0:
        distances /= diag

    # ヒストグラム
    hist, _ = np.histogram(distances, bins=bins, range=(0, 2.0), density=True)
    return hist


def trimesh_sample_surface(mesh, count: int):
    """trimesh の表面サンプリング."""
    return mesh.sample(count, return_index=True)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
