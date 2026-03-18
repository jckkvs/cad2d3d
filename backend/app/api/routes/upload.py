"""
ファイルアップロード API.

画像・CADファイルのアップロード処理を担当.

Implements: F-001 (多形式ファイル入力)
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.config import settings
from app.models.schemas import APIResponse, UploadedFileInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


def _get_allowed_extensions() -> set[str]:
    """許可される拡張子の集合を返す."""
    return set(
        settings.allowed_image_extensions
        + settings.allowed_cad_extensions
        + settings.allowed_document_extensions
    )


@router.post("/", response_model=APIResponse)
async def upload_files(files: list[UploadFile] = File(...)) -> APIResponse:
    """
    1つ以上のファイルをアップロード.

    対応: JPG, PNG, BMP, TIFF, PDF, DXF, SVG, HEIC 等.
    """
    settings.ensure_dirs()
    allowed = _get_allowed_extensions()
    uploaded: list[dict] = []

    for file in files:
        if not file.filename:
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"サポートされていないファイル形式です: {ext}. "
                f"対応形式: {', '.join(sorted(allowed))}",
            )

        # ファイルサイズチェック
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"ファイルサイズが上限を超えています: "
                f"{size_mb:.1f} MB > {settings.max_upload_size_mb} MB",
            )

        # 保存
        file_id = str(uuid.uuid4())
        stored_name = f"{file_id}{ext}"
        stored_path = settings.upload_dir / stored_name
        stored_path.write_bytes(content)

        info = UploadedFileInfo(
            id=file_id,
            original_name=file.filename,
            stored_path=str(stored_path),
            file_size=len(content),
            mime_type=file.content_type,
        )
        uploaded.append(info.model_dump(mode="json"))
        logger.info("Uploaded file: %s -> %s", file.filename, stored_path)

    return APIResponse(
        success=True,
        message=f"{len(uploaded)} ファイルをアップロードしました。",
        data=uploaded,
    )


@router.delete("/{file_id}", response_model=APIResponse)
async def delete_file(file_id: str) -> APIResponse:
    """アップロード済みファイルを削除."""
    # ファイル ID から検索
    for candidate in settings.upload_dir.iterdir():
        if candidate.stem == file_id:
            candidate.unlink()
            logger.info("Deleted file: %s", candidate)
            return APIResponse(success=True, message="ファイルを削除しました。")

    raise HTTPException(status_code=404, detail="ファイルが見つかりません。")
