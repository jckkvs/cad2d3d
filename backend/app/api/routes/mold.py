"""
金型設計 API ルート.

金型構造推定、アンダーカット検出、部品DB、類似度比較のAPIエンドポイント.

Implements: F-070〜F-081
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.core.config import settings
from app.models.schemas import APIResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mold", tags=["mold"])


# ── リクエスト/レスポンススキーマ ──

class MoldDesignRequest(BaseModel):
    mesh_path: str
    parting_direction: list[float] | None = None
    cavity_count: int = 1


class UndercutRequest(BaseModel):
    mesh_path: str
    parting_direction: list[float] | None = None
    ray_density: int = 50


class SimilarityRequest(BaseModel):
    path_a: str
    path_b: str


class PartAddRequest(BaseModel):
    id: str
    category: str
    name: str
    manufacturer: str
    model_number: str
    diameter: float | None = None
    length: float | None = None
    material: str = ""
    unit_price: float = 0.0
    notes: str = ""


# ── エンドポイント ──

@router.post("/design", response_model=APIResponse)
async def design_mold(req: MoldDesignRequest) -> APIResponse:
    """3DメッシュからDL金型構造を推定."""
    from app.mold.core import MoldDesigner
    import numpy as np

    mesh_path = Path(req.mesh_path)
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="メッシュファイルが見つかりません。")

    direction = np.array(req.parting_direction) if req.parting_direction else None

    try:
        designer = MoldDesigner()
        result = designer.design(mesh_path, direction, req.cavity_count)

        return APIResponse(
            success=True,
            message=result.summary,
            data={
                "total_size": {
                    "width": result.total_width,
                    "height": result.total_height,
                    "depth": result.total_depth,
                },
                "total_weight_kg": result.total_weight_kg,
                "components_count": len(result.components),
                "components": [
                    {"name": c.name, "quantity": c.quantity, "size": c.size, "material": c.material}
                    for c in result.components
                ],
                "has_undercut": result.undercut_result.has_undercut if result.undercut_result else False,
                "undercut_count": result.undercut_result.undercut_count if result.undercut_result else 0,
                "estimated_cost": result.estimated_total_cost,
            },
        )
    except Exception as e:
        logger.error("Mold design failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"金型設計推定エラー: {e}") from e


@router.post("/undercut", response_model=APIResponse)
async def analyze_undercut(req: UndercutRequest) -> APIResponse:
    """アンダーカット解析を実行."""
    from app.mold.undercut import detect_undercuts
    import numpy as np

    mesh_path = Path(req.mesh_path)
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="メッシュファイルが見つかりません。")

    direction = np.array(req.parting_direction) if req.parting_direction else None

    try:
        result = detect_undercuts(mesh_path, direction, req.ray_density)
        return APIResponse(
            success=True,
            message=result.summary,
            data={
                "has_undercut": result.has_undercut,
                "undercut_count": result.undercut_count,
                "total_undercut_area": result.total_undercut_area,
                "parting_direction": result.parting_direction.tolist(),
                "regions": [
                    {
                        "area": r.area,
                        "depth": r.depth,
                        "severity": r.severity,
                        "mechanism": r.recommended_mechanism,
                        "centroid": r.centroid.tolist(),
                    }
                    for r in result.undercut_regions
                ],
                "core_faces_count": len(result.core_faces),
                "cavity_faces_count": len(result.cavity_faces),
            },
        )
    except Exception as e:
        logger.error("Undercut analysis failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"アンダーカット解析エラー: {e}") from e


@router.post("/parting-line", response_model=APIResponse)
async def optimize_parting(req: UndercutRequest) -> APIResponse:
    """パーティングラインを最適化."""
    from app.mold.parting_line import optimize_parting_line

    mesh_path = Path(req.mesh_path)
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="メッシュファイルが見つかりません。")

    try:
        result = optimize_parting_line(mesh_path)
        return APIResponse(
            success=True,
            message=result.summary,
            data={
                "best_direction": result.best.direction.tolist(),
                "best_score": result.best.score,
                "mold_height": result.best.mold_height,
                "mold_volume": result.best.mold_volume,
                "undercut_area": result.best.undercut_area,
                "candidates": [
                    {
                        "direction": c.direction.tolist(),
                        "score": c.score,
                        "mold_height": c.mold_height,
                        "undercut_area": c.undercut_area,
                    }
                    for c in result.candidates[:6]
                ],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"パーティングライン最適化エラー: {e}") from e


# ── 部品DB ──

@router.get("/parts/categories")
async def list_part_categories() -> dict:
    """部品カテゴリ一覧."""
    from app.mold.parts_db import parts_db
    return parts_db.list_categories()


@router.get("/parts")
async def list_parts(
    category: str | None = None,
    min_diameter: float | None = None,
    min_length: float | None = None,
) -> list[dict]:
    """部品検索."""
    from app.mold.parts_db import parts_db
    from dataclasses import asdict
    results = parts_db.search(category=category, min_diameter=min_diameter, min_length=min_length)
    return [asdict(p) for p in results]


@router.post("/parts")
async def add_part(req: PartAddRequest) -> APIResponse:
    """部品を追加."""
    from app.mold.parts_db import parts_db, MoldPart
    part = MoldPart(
        id=req.id, category=req.category, name=req.name,
        manufacturer=req.manufacturer, model_number=req.model_number,
        diameter=req.diameter, length=req.length,
        material=req.material, unit_price=req.unit_price, notes=req.notes,
    )
    parts_db.add_part(part)
    return APIResponse(success=True, message=f"部品 {req.id} を追加しました。")


@router.get("/parts/recommend")
async def recommend_part(
    category: str,
    diameter: float | None = None,
    length: float | None = None,
) -> dict | None:
    """必要サイズに基づく部品推奨."""
    from app.mold.parts_db import parts_db
    from dataclasses import asdict
    part = parts_db.recommend(category, diameter, length)
    return asdict(part) if part else None


# ── 類似度比較 ──

@router.post("/similarity/mesh", response_model=APIResponse)
async def compare_meshes_api(req: SimilarityRequest) -> APIResponse:
    """3Dメッシュの形状類似度比較."""
    from app.similarity.mesh_similarity import compare_mesh_files

    path_a, path_b = Path(req.path_a), Path(req.path_b)
    if not path_a.exists() or not path_b.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません。")

    try:
        result = compare_mesh_files(path_a, path_b)
        return APIResponse(
            success=True,
            message=f"類似度スコア: {result.score:.1%}",
            data={
                "score": result.score,
                "d2_similarity": result.d2_similarity,
                "aspect_similarity": result.aspect_similarity,
                "volume_similarity": result.volume_similarity,
                "compactness_similarity": result.compactness_similarity,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"メッシュ比較エラー: {e}") from e


@router.post("/similarity/image", response_model=APIResponse)
async def compare_images_api(req: SimilarityRequest) -> APIResponse:
    """2D画像の類似度比較."""
    from app.similarity.image_similarity import compare_image_files

    path_a, path_b = Path(req.path_a), Path(req.path_b)
    if not path_a.exists() or not path_b.exists():
        raise HTTPException(status_code=404, detail="ファイルが見つかりません。")

    try:
        result = compare_image_files(path_a, path_b)
        return APIResponse(
            success=True,
            message=f"類似度スコア: {result.score:.1%}",
            data={
                "score": result.score,
                "histogram_similarity": result.histogram_similarity,
                "edge_similarity": result.edge_similarity,
                "aspect_similarity": result.aspect_similarity,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"画像比較エラー: {e}") from e


# ── サイジング ──

class ClampForceRequest(BaseModel):
    projected_area_mm2: float
    resin: str = "ABS"
    cavity_count: int = 1
    safety_factor: float = 1.2


class CycleTimeRequest(BaseModel):
    wall_thickness_mm: float
    part_weight_g: float
    resin: str = "ABS"
    production_quantity: int = 10000
    cavity_count: int = 1


@router.post("/sizing/clamp-force", response_model=APIResponse)
async def calc_clamp_force(req: ClampForceRequest) -> APIResponse:
    """型締力を計算."""
    from app.mold.sizing import calculate_clamp_force
    result = calculate_clamp_force(
        req.projected_area_mm2, req.resin, req.cavity_count, req.safety_factor
    )
    return APIResponse(
        success=True,
        message=f"推奨成形機: {result.recommended_machine}",
        data={
            "projected_area_cm2": result.projected_area_cm2,
            "cavity_pressure_mpa": result.cavity_pressure_mpa,
            "clamp_force_kn": result.clamp_force_kn,
            "clamp_force_ton": result.clamp_force_ton,
            "recommended_machine": result.recommended_machine,
        },
    )


@router.post("/sizing/cycle-time", response_model=APIResponse)
async def calc_cycle_time(req: CycleTimeRequest) -> APIResponse:
    """サイクルタイムを概算."""
    from app.mold.sizing import estimate_cycle_time
    result = estimate_cycle_time(
        req.wall_thickness_mm, req.part_weight_g, req.resin,
        req.production_quantity, req.cavity_count,
    )
    return APIResponse(
        success=True,
        message=f"サイクル: {result.total_cycle_s:.1f}秒, {result.shots_per_hour}ショット/h",
        data={
            "injection_time_s": result.injection_time_s,
            "hold_time_s": result.hold_time_s,
            "cooling_time_s": result.cooling_time_s,
            "mold_open_time_s": result.mold_open_time_s,
            "total_cycle_s": result.total_cycle_s,
            "shots_per_hour": result.shots_per_hour,
            "production_time_hours": result.production_time_hours,
        },
    )


@router.get("/resins")
async def list_resins() -> dict[str, str]:
    """使用可能な樹脂一覧."""
    from app.mold.sizing import RESIN_DB
    return {k: v.name for k, v in RESIN_DB.items()}


@router.post("/draft-analysis", response_model=APIResponse)
async def analyze_draft(req: UndercutRequest) -> APIResponse:
    """ドラフト角を解析."""
    from app.mold.draft_analysis import analyze_draft_angles
    import numpy as np

    mesh_path = Path(req.mesh_path)
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail="メッシュファイルが見つかりません。")

    direction = np.array(req.parting_direction) if req.parting_direction else None

    try:
        result = analyze_draft_angles(mesh_path, direction)
        return APIResponse(
            success=True,
            message=result.summary,
            data={
                "compliance_ratio": result.compliance_ratio,
                "average_draft_deg": result.average_draft_deg,
                "min_draft_deg": result.min_draft_deg,
                "max_draft_deg": result.max_draft_deg,
                "total_faces": len(result.faces),
                "insufficient_count": len(result.insufficient_faces),
                "insufficient_area": result.total_insufficient_area,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ドラフト角解析エラー: {e}") from e
