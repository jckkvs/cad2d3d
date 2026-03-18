"""
設定 API.

プロキシ・HuggingFaceトークン等のアプリ設定を管理.

Implements: F-021〜F-022
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import APIResponse, AppSettings, ProxySettings, HuggingFaceSettings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("/", response_model=APIResponse)
async def get_settings() -> APIResponse:
    """現在の設定を取得."""
    current = AppSettings(
        proxy=ProxySettings(
            http_proxy=settings.http_proxy,
            https_proxy=settings.https_proxy,
        ),
        huggingface=HuggingFaceSettings(
            token="***" if settings.hf_token else None,  # トークンはマスク
            cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
        ),
    )
    return APIResponse(success=True, data=current.model_dump())


@router.put("/proxy", response_model=APIResponse)
async def update_proxy(proxy: ProxySettings) -> APIResponse:
    """プロキシ設定を更新."""
    settings.http_proxy = proxy.http_proxy
    settings.https_proxy = proxy.https_proxy
    logger.info("Proxy settings updated: http=%s, https=%s", proxy.http_proxy, proxy.https_proxy)
    return APIResponse(success=True, message="プロキシ設定を更新しました。")


@router.put("/huggingface", response_model=APIResponse)
async def update_hf_settings(hf: HuggingFaceSettings) -> APIResponse:
    """HuggingFace設定を更新."""
    if hf.token and hf.token != "***":
        settings.hf_token = hf.token
    if hf.cache_dir:
        settings.hf_cache_dir = Path(hf.cache_dir)
    logger.info("HuggingFace settings updated")
    return APIResponse(success=True, message="HuggingFace設定を更新しました。")
