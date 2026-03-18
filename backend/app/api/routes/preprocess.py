"""
前処理 API.

画像前処理パイプラインのREST APIインターフェース.

Implements: F-040〜F-043
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.models.schemas import APIResponse
from app.preprocessing.pipeline import (
    PreprocessingConfig,
    preprocess_file,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/preprocess", tags=["preprocess"])


@router.post("/{file_id}", response_model=APIResponse)
async def preprocess_uploaded_file(
    file_id: str,
    auto_split: bool = True,
    remove_text: bool = True,
    remove_dimensions: bool = True,
    remove_hatching: bool = True,
    remove_auxiliary: bool = True,
) -> APIResponse:
    """
    アップロード済みファイルの前処理を実行.

    自動でフォーマット変換 → マルチビュー分割 → 注釈除去を行う.
    """
    # ファイル検索
    input_path: Path | None = None
    for candidate in settings.upload_dir.iterdir():
        if candidate.stem == file_id:
            input_path = candidate
            break

    if not input_path:
        raise HTTPException(status_code=404, detail="ファイルが見つかりません。")

    config = PreprocessingConfig(
        auto_split_views=auto_split,
        remove_text=remove_text,
        remove_dimensions=remove_dimensions,
        remove_hatching=remove_hatching,
        remove_auxiliary=remove_auxiliary,
    )

    output_dir = settings.temp_dir / f"preprocess_{file_id}"

    try:
        results = preprocess_file(input_path, output_dir, config)

        response_data = []
        for r in results:
            response_data.append({
                "path": str(r.path),
                "original_path": str(r.original_path),
                "view_label": r.view_label,
                "was_split": r.was_split,
                "scale_info": {
                    "value": r.scale_info.value,
                    "unit": r.scale_info.unit,
                    "source_text": r.scale_info.source_text,
                } if r.scale_info else None,
                "annotations_removed": len(r.removed_annotations),
            })

        return APIResponse(
            success=True,
            message=f"{len(results)} 枚の前処理済み画像を生成しました。",
            data=response_data,
        )

    except Exception as e:
        logger.error("Preprocessing failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"前処理に失敗: {e}") from e
