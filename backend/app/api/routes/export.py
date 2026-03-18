"""
エクスポート API.

3Dモデルのフォーマット変換・ダウンロード・外部CADソフトでの起動を担当.

Implements: F-030〜F-034 (3Dビューア・編集)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.schemas import APIResponse, OutputFormat

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/export", tags=["export"])


@router.get("/download/{job_id}")
async def download_result(job_id: str, format: str | None = None) -> FileResponse:
    """
    生成結果ファイルをダウンロード.

    format パラメータで出力形式を指定可能 (変換はPhase 2以降)。
    """
    output_dir = settings.temp_dir / job_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="ジョブ結果が見つかりません。")

    # 出力ディレクトリ内のファイルを探す
    output_files = list(output_dir.iterdir())
    if not output_files:
        raise HTTPException(status_code=404, detail="出力ファイルが見つかりません。")

    # 指定フォーマットのフィルタリング
    target_file = output_files[0]
    if format:
        for f in output_files:
            if f.suffix.lstrip(".").lower() == format.lower():
                target_file = f
                break

    return FileResponse(
        path=str(target_file),
        filename=target_file.name,
        media_type="application/octet-stream",
    )


@router.post("/open-external/{job_id}", response_model=APIResponse)
async def open_in_external_app(job_id: str) -> APIResponse:
    """
    外部CADソフトで3Dモデルを開く.

    OSのファイル関連付けを利用して適切なアプリで開く.
    """
    output_dir = settings.temp_dir / job_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="ジョブ結果が見つかりません。")

    output_files = list(output_dir.iterdir())
    if not output_files:
        raise HTTPException(status_code=404, detail="出力ファイルが見つかりません。")

    target_file = output_files[0]

    try:
        if sys.platform == "win32":
            os.startfile(str(target_file))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(target_file)], check=True)
        else:
            subprocess.run(["xdg-open", str(target_file)], check=True)

        return APIResponse(
            success=True,
            message=f"'{target_file.name}' を外部アプリケーションで開きました。",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"外部アプリでの起動に失敗しました: {e}",
        ) from e


@router.post("/reload/{job_id}", response_model=APIResponse)
async def reload_from_external(job_id: str) -> APIResponse:
    """
    外部編集後のCADデータを再読み込み.

    外部CADソフトで編集されたファイルの変更を反映.
    """
    output_dir = settings.temp_dir / job_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="ジョブ結果が見つかりません。")

    output_files = list(output_dir.iterdir())
    if not output_files:
        raise HTTPException(status_code=404, detail="出力ファイルが見つかりません。")

    target_file = output_files[0]

    return APIResponse(
        success=True,
        message=f"'{target_file.name}' を再読み込みしました。",
        data={
            "file_path": str(target_file),
            "file_size": target_file.stat().st_size,
            "last_modified": target_file.stat().st_mtime,
        },
    )


@router.get("/formats", response_model=APIResponse)
async def list_output_formats() -> APIResponse:
    """利用可能な出力フォーマット一覧."""
    formats = [{"value": f.value, "label": f.value.upper()} for f in OutputFormat]
    return APIResponse(success=True, data=formats)
