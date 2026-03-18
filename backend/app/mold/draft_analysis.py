"""
ドラフト角（抜き勾配）解析モジュール.

金型から製品を離型するために必要な抜き勾配を解析する.
各面のドラフト角を計算し、不足を警告する.

参考文献:
- "Design of Injection Molds" (Gastrow)
  原文: "Draft angles of at least 0.5° to 1° per side are usually
  recommended for smooth ejection."
  訳: スムーズな離型のために片側0.5°〜1°以上のドラフト角が推奨.

- "Injection Mold Design Engineering" (Kazmer)
  原文: "Insufficient draft can result in part sticking, surface
  scratching, and increased ejection force."
  訳: 不十分なドラフト角は製品の固着・表面傷・突出力増大を引き起こす.

Implements: F-075 (ドラフト角解析)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.exceptions import CAD3DError

logger = logging.getLogger(__name__)


@dataclass
class DraftAngleInfo:
    """各面のドラフト角情報."""
    face_index: int
    draft_angle_deg: float      # ドラフト角 [度]
    face_area: float            # 面積 [mm²]
    is_sufficient: bool         # 十分か
    face_normal: np.ndarray     # 法線


@dataclass
class DraftAnalysisResult:
    """ドラフト角解析の結果."""
    parting_direction: np.ndarray
    min_required_draft_deg: float   # 最小要求ドラフト角
    faces: list[DraftAngleInfo] = field(default_factory=list)
    insufficient_faces: list[DraftAngleInfo] = field(default_factory=list)
    average_draft_deg: float = 0.0
    min_draft_deg: float = 0.0
    max_draft_deg: float = 0.0
    total_insufficient_area: float = 0.0
    compliance_ratio: float = 0.0   # 合格率
    summary: str = ""


def analyze_draft_angles(
    mesh_path: Path,
    parting_direction: np.ndarray | None = None,
    min_draft_deg: float = 1.0,
) -> DraftAnalysisResult:
    """
    3Dメッシュの全面についてドラフト角を解析.

    ドラフト角 = 90° - arccos(|n · d|)
    ここで n=面法線, d=型開き方向.

    Args:
        mesh_path: メッシュファイルパス.
        parting_direction: 型開き方向 (None=Z軸).
        min_draft_deg: 最小要求ドラフト角 [度].

    Returns:
        ドラフト角解析結果.
    """
    try:
        import trimesh
    except ImportError:
        raise CAD3DError("trimesh が必要です")

    mesh = trimesh.load(str(mesh_path))
    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise CAD3DError("メッシュが見つかりません")
        mesh = max(geometries, key=lambda g: len(g.faces) if hasattr(g, 'faces') else 0)

    if parting_direction is None:
        parting_direction = np.array([0, 0, 1], dtype=np.float64)
    else:
        parting_direction = np.asarray(parting_direction, dtype=np.float64)
        parting_direction = parting_direction / np.linalg.norm(parting_direction)

    face_normals = mesh.face_normals
    face_areas = mesh.area_faces

    # 各面のドラフト角を計算
    dots = np.abs(np.dot(face_normals, parting_direction))
    # パーティング面 (法線が型開き方向に垂直) → ドラフト角 = 0
    # 面が型開き方向に平行 → ドラフト角 = 90° (完全OK)
    # ドラフト角 = 90° - arccos(|dot|) のうち側面を評価
    # 側面 = dotが0に近い面 → ドラフト角 = 90° - 90° = 0 (不足)
    # 実質: ドラフト角 = arcsin(|dot|) (dotが小さい面がドラフト不足)

    # ただし、天面・底面(dot≈1)はドラフト対象外
    # 側面のみ(dot < cos(5°) ≈ 0.996)を評価対象とする
    parting_threshold = np.cos(np.radians(5.0))  # 天面/底面の閾値

    faces_info: list[DraftAngleInfo] = []
    insufficient: list[DraftAngleInfo] = []

    for i in range(len(face_normals)):
        dot_val = dots[i]

        # 天面/底面は除外 (型開き方向にほぼ平行な法線)
        if dot_val > parting_threshold:
            continue

        # ドラフト角 = arcsin(|dot|) [度]
        draft_deg = float(np.degrees(np.arcsin(min(dot_val, 1.0))))
        is_ok = draft_deg >= min_draft_deg

        info = DraftAngleInfo(
            face_index=i,
            draft_angle_deg=draft_deg,
            face_area=float(face_areas[i]),
            is_sufficient=is_ok,
            face_normal=face_normals[i],
        )
        faces_info.append(info)
        if not is_ok:
            insufficient.append(info)

    # 統計
    if faces_info:
        drafts = [f.draft_angle_deg for f in faces_info]
        avg_draft = float(np.mean(drafts))
        min_d = float(np.min(drafts))
        max_d = float(np.max(drafts))
        compliance = 1.0 - len(insufficient) / len(faces_info)
    else:
        avg_draft = 0.0
        min_d = 0.0
        max_d = 0.0
        compliance = 1.0

    insuf_area = sum(f.face_area for f in insufficient)

    result = DraftAnalysisResult(
        parting_direction=parting_direction,
        min_required_draft_deg=min_draft_deg,
        faces=faces_info,
        insufficient_faces=insufficient,
        average_draft_deg=avg_draft,
        min_draft_deg=min_d,
        max_draft_deg=max_d,
        total_insufficient_area=insuf_area,
        compliance_ratio=compliance,
    )

    # サマリ
    lines = [
        f"▼ ドラフト角解析",
        f"  要求ドラフト角: ≥{min_draft_deg}°",
        f"  評価面数: {len(faces_info)}",
        f"  不足面数: {len(insufficient)} (面積: {insuf_area:.1f} mm²)",
        f"  合格率: {compliance:.0%}",
        f"  平均ドラフト角: {avg_draft:.1f}°",
        f"  最小: {min_d:.1f}°, 最大: {max_d:.1f}°",
    ]
    if insufficient:
        lines.append("  ⚠ ドラフト角不足の面があります。離型困難の可能性があります。")
    else:
        lines.append("  ✓ 全面で十分なドラフト角が確保されています。")
    result.summary = "\n".join(lines)

    logger.info("Draft analysis: %d faces evaluated, %d insufficient", len(faces_info), len(insufficient))
    return result
