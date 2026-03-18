"""
CAD3D Generator — FastAPI メインアプリケーション.

2D画像/CADから3D CADモデルを生成するWebアプリケーション.
すべての処理はローカルで実行され、画像データは外部に送信されない.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.exceptions import CAD3DError
from app.engines.registry import EngineRegistry

# ルートのインポート
from app.api.routes import upload, generate, models, export, preprocess, mold
from app.api.routes import settings as settings_route

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """アプリ起動/終了時のライフサイクル管理."""
    # 起動時
    logger.info("=== CAD3D Generator v%s starting ===", settings.app_version)
    settings.ensure_dirs()
    EngineRegistry.discover_engines()
    engines = EngineRegistry.list_available()
    logger.info("Available engines: %s", [e.name for e in engines])
    yield
    # 終了時
    logger.info("=== CAD3D Generator shutting down ===")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="2D画像/CADから3D CADモデルを生成するローカルWebアプリケーション",
    lifespan=lifespan,
)

# CORS設定 (フロントエンドからのアクセスを許可)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIルートの登録
app.include_router(upload.router, prefix="/api")
app.include_router(generate.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(export.router, prefix="/api")
app.include_router(preprocess.router, prefix="/api")
app.include_router(mold.router, prefix="/api")
app.include_router(settings_route.router, prefix="/api")


# カスタム例外ハンドラ
@app.exception_handler(CAD3DError)
async def cad3d_error_handler(request: Request, exc: CAD3DError) -> JSONResponse:
    """CAD3DError系例外のハンドリング."""
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "message": exc.message,
            "detail": exc.detail,
        },
    )


# 静的ファイル (フロントエンドビルド成果物)
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


@app.get("/api/health")
async def health_check() -> dict:
    """ヘルスチェック."""
    return {
        "status": "ok",
        "version": settings.app_version,
        "engines_count": len(EngineRegistry.list_available()),
    }
