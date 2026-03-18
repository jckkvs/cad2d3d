"""
モデル重み管理 API.

重みのダウンロード・ステータス確認・手動配置ガイドを担当.

Implements: F-020〜F-024
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from app.engines.registry import EngineRegistry
from app.core.exceptions import EngineNotFoundError, WeightDownloadError
from app.models.schemas import APIResponse
from app.weights.manager import weight_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/weights", tags=["weights"])


@router.get("/{engine_name}", response_model=APIResponse)
async def get_weight_status(engine_name: str) -> APIResponse:
    """指定エンジンの重みファイルステータスを返す."""
    try:
        engine = EngineRegistry.get(engine_name)
    except EngineNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message) from e

    weights = engine.get_required_weights()
    status = weight_manager.get_all_status(weights)

    return APIResponse(
        success=True,
        data=status,
    )


@router.post("/{engine_name}/download", response_model=APIResponse)
async def download_weights(engine_name: str) -> APIResponse:
    """指定エンジンの全ての重みファイルをダウンロード."""
    try:
        engine = EngineRegistry.get(engine_name)
    except EngineNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message) from e

    weights = engine.get_required_weights()
    results = []

    for w in weights:
        if weight_manager.check_weight_exists(w):
            results.append({"name": w.name, "status": "already_exists"})
            continue

        try:
            await weight_manager.download_weight(w)
            results.append({"name": w.name, "status": "downloaded"})
        except WeightDownloadError as e:
            results.append({"name": w.name, "status": "failed", "error": e.message})

    all_ok = all(r["status"] != "failed" for r in results)
    return APIResponse(
        success=all_ok,
        message="重みファイルのダウンロードが完了しました。" if all_ok else "一部のダウンロードに失敗しました。",
        data=results,
    )


@router.post("/{engine_name}/download/{weight_index}", response_model=APIResponse)
async def download_single_weight(engine_name: str, weight_index: int) -> APIResponse:
    """指定した重みファイルを個別にダウンロード."""
    try:
        engine = EngineRegistry.get(engine_name)
    except EngineNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message) from e

    weights = engine.get_required_weights()
    if weight_index < 0 or weight_index >= len(weights):
        raise HTTPException(status_code=400, detail="無効なインデックスです。")

    w = weights[weight_index]

    if weight_manager.check_weight_exists(w):
        return APIResponse(success=True, message="既にダウンロード済みです。")

    try:
        await weight_manager.download_weight(w)
        return APIResponse(success=True, message=f"'{w.name}' のダウンロードが完了しました。")
    except WeightDownloadError as e:
        raise HTTPException(status_code=500, detail=e.message) from e
