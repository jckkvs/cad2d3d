"""
3D生成 API.

エンジン選択・3D生成ジョブの実行・進捗管理を担当.

Implements: F-010〜F-013 (3D生成)
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.exceptions import EngineNotFoundError, EngineNotReadyError
from app.engines.base import ProcessedImage, ReconstructionParams
from app.engines.registry import EngineRegistry
from app.models.schemas import (
    APIResponse,
    EngineInfo,
    EngineStatus,
    GenerationProgress,
    GenerationRequest,
    GenerationResult,
    JobStatus,
    OutputFormat,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/generate", tags=["generate"])

# ジョブ管理 (インメモリ; 軽量設計)
_jobs: dict[str, GenerationProgress] = {}
_results: dict[str, GenerationResult] = {}


@router.get("/engines", response_model=list[EngineInfo])
async def list_engines() -> list[EngineInfo]:
    """利用可能なエンジン一覧を返す."""
    return EngineRegistry.list_available()


@router.get("/engines/{engine_name}", response_model=EngineInfo)
async def get_engine_info(engine_name: str) -> EngineInfo:
    """指定エンジンの詳細情報を返す."""
    try:
        engine = EngineRegistry.get(engine_name)
        return engine.get_info()
    except EngineNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message) from e


@router.get("/engines/{engine_name}/readme")
async def get_engine_readme(engine_name: str) -> APIResponse:
    """エンジンのREADME_MODEL.mdの内容を返す."""
    try:
        engine = EngineRegistry.get(engine_name)
        info = engine.get_info()
        if info.readme_path and Path(info.readme_path).exists():
            content = Path(info.readme_path).read_text(encoding="utf-8")
            return APIResponse(success=True, data={"content": content})
        return APIResponse(success=False, message="README_MODEL.md が見つかりません。")
    except EngineNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message) from e


@router.post("/run", response_model=APIResponse)
async def start_generation(request: GenerationRequest) -> APIResponse:
    """
    3D生成ジョブを開始.

    非同期で実行し、ジョブIDを返す。進捗はWebSocketで通知。
    """
    try:
        engine = EngineRegistry.get(request.engine_name)
    except EngineNotFoundError as e:
        raise HTTPException(status_code=404, detail=e.message) from e

    # エンジン準備チェック
    status = engine.check_ready()
    if status != EngineStatus.READY:
        raise HTTPException(
            status_code=503,
            detail=f"エンジン '{request.engine_name}' の準備ができていません: {status.value}",
        )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = GenerationProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
        progress=0.0,
        message="ジョブを開始します...",
    )

    # バックグラウンドで生成を実行
    asyncio.create_task(_run_job(job_id, engine, request))

    return APIResponse(
        success=True,
        message="3D生成ジョブを開始しました。",
        data={"job_id": job_id},
    )


@router.get("/jobs/{job_id}", response_model=GenerationProgress)
async def get_job_status(job_id: str) -> GenerationProgress:
    """ジョブの進捗を取得."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません。")
    return _jobs[job_id]


@router.get("/jobs/{job_id}/result", response_model=GenerationResult)
async def get_job_result(job_id: str) -> GenerationResult:
    """ジョブの結果を取得."""
    if job_id not in _results:
        if job_id in _jobs and _jobs[job_id].status not in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
        ):
            raise HTTPException(status_code=202, detail="ジョブはまだ完了していません。")
        raise HTTPException(status_code=404, detail="ジョブ結果が見つかりません。")
    return _results[job_id]


# ── WebSocket (進捗通知) ─────────────────────────────────


@router.websocket("/ws/{job_id}")
async def generation_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """ジョブの進捗をWebSocketでリアルタイム通知."""
    await websocket.accept()
    try:
        while True:
            if job_id in _jobs:
                progress = _jobs[job_id]
                await websocket.send_json(progress.model_dump(mode="json"))
                if progress.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)
    except Exception:
        logger.warning("WebSocket error for job %s", job_id, exc_info=True)


# ── 内部ジョブ実行 ───────────────────────────────────────


async def _run_job(
    job_id: str,
    engine: Any,
    request: GenerationRequest,
) -> None:
    """バックグラウンドで3D生成ジョブを実行."""
    start_time = time.time()

    def progress_cb(progress: float, message: str) -> None:
        _jobs[job_id] = GenerationProgress(
            job_id=job_id,
            status=JobStatus.GENERATING,
            progress=progress,
            message=message,
        )

    try:
        _jobs[job_id].status = JobStatus.PREPROCESSING

        # 画像パスを構築
        images = []
        for img in request.images:
            # upload_dir から探す
            for candidate in settings.upload_dir.iterdir():
                if candidate.stem == img.file_id:
                    images.append(
                        ProcessedImage(
                            path=candidate,
                            view_angle=img.view_angle.value,
                            azimuth=img.custom_azimuth,
                            elevation=img.custom_elevation,
                        )
                    )
                    break

        if not images:
            raise ValueError("アップロードされた画像が見つかりません。")

        _jobs[job_id].status = JobStatus.GENERATING

        # 出力ディレクトリ
        output_dir = settings.temp_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        params = ReconstructionParams(
            output_format=request.output_format,
            output_dir=output_dir,
            extra=request.engine_params,
        )

        result = await engine.reconstruct(images, params, progress_cb)

        elapsed = time.time() - start_time

        if result.success:
            _jobs[job_id] = GenerationProgress(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                progress=1.0,
                message="生成完了!",
            )
            _results[job_id] = GenerationResult(
                job_id=job_id,
                status=JobStatus.COMPLETED,
                output_file=str(result.output_path) if result.output_path else None,
                output_format=result.output_format,
                elapsed_seconds=elapsed,
                engine_name=request.engine_name,
                metadata=result.metadata,
            )
        else:
            _jobs[job_id] = GenerationProgress(
                job_id=job_id,
                status=JobStatus.FAILED,
                progress=0.0,
                message="生成失敗",
                error=result.error_message,
            )

    except Exception as e:
        logger.error("Job %s failed", job_id, exc_info=True)
        _jobs[job_id] = GenerationProgress(
            job_id=job_id,
            status=JobStatus.FAILED,
            progress=0.0,
            message="エラーが発生しました",
            error=str(e),
        )


# ── 比較生成 API ──────────────────────────────────────────


@router.post("/compare", response_model=APIResponse)
async def compare_generation(request: GenerationRequest) -> APIResponse:
    """
    複数エンジンで並列生成し、結果を比較.

    engine_params に "compare_engines" キーで比較対象エンジン名のリストを渡す.
    例: {"compare_engines": ["triposr", "instantmesh", "crm"]}
    """
    compare_engines = request.engine_params.get("compare_engines", [])
    if not compare_engines:
        compare_engines = [request.engine_name]

    # 利用可能なエンジンのみフィルタ
    available = {e.name for e in EngineRegistry.list_available()}
    valid_engines = [e for e in compare_engines if e in available]

    if not valid_engines:
        raise HTTPException(
            status_code=404,
            detail=f"有効なエンジンが見つかりません。利用可能: {sorted(available)}",
        )

    # 各エンジンのジョブを開始
    job_ids: dict[str, str] = {}
    for eng_name in valid_engines:
        try:
            engine = EngineRegistry.get(eng_name)
            if engine.check_ready() != EngineStatus.READY:
                continue

            job_id = str(uuid.uuid4())
            _jobs[job_id] = GenerationProgress(
                job_id=job_id,
                status=JobStatus.QUEUED,
                progress=0.0,
                message=f"{eng_name}: ジョブ開始...",
            )

            # 各エンジン用のリクエストを作成
            eng_request = GenerationRequest(
                engine_name=eng_name,
                images=request.images,
                output_format=request.output_format,
                engine_params=request.engine_params,
            )
            asyncio.create_task(_run_job(job_id, engine, eng_request))
            job_ids[eng_name] = job_id

        except Exception as e:
            logger.warning("Engine %s failed to start: %s", eng_name, e)

    return APIResponse(
        success=True,
        message=f"{len(job_ids)}エンジンで比較生成を開始しました。",
        data={
            "comparison_jobs": job_ids,
            "engines": list(job_ids.keys()),
        },
    )


@router.get("/compare/{job_ids_csv}", response_model=APIResponse)
async def get_comparison_results(job_ids_csv: str) -> APIResponse:
    """比較生成の結果を一覧取得."""
    job_ids = job_ids_csv.split(",")
    results: list[dict[str, Any]] = []

    for job_id in job_ids:
        entry: dict[str, Any] = {"job_id": job_id.strip()}
        jid = job_id.strip()

        if jid in _results:
            r = _results[jid]
            entry["status"] = "completed"
            entry["engine"] = r.engine_name
            entry["elapsed"] = r.elapsed_seconds
            entry["output_file"] = r.output_file
            entry["metadata"] = r.metadata
        elif jid in _jobs:
            j = _jobs[jid]
            entry["status"] = j.status.value
            entry["progress"] = j.progress
            entry["message"] = j.message
        else:
            entry["status"] = "not_found"

        results.append(entry)

    all_done = all(r.get("status") in ("completed", "failed", "not_found") for r in results)

    return APIResponse(
        success=True,
        message="完了" if all_done else "生成中...",
        data={"results": results, "all_done": all_done},
    )
