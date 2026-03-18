"""
モデル重みファイルの管理.

ダウンロード・プロキシ対応・HuggingFace トークン管理を担当.

Implements: F-020〜F-024 (重み管理)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable

import httpx

from app.core.config import settings
from app.core.exceptions import WeightDownloadError
from app.models.schemas import WeightFileInfo

logger = logging.getLogger(__name__)


class WeightManager:
    """モデル重みファイルのダウンロード・管理."""

    def __init__(self) -> None:
        self._active_downloads: dict[str, float] = {}  # relative_path -> progress

    def check_weight_exists(self, weight: WeightFileInfo) -> bool:
        """指定の重みファイルが存在するか確認."""
        path = settings.weights_dir / weight.relative_path
        return path.exists()

    def get_weight_path(self, weight: WeightFileInfo) -> Path:
        """重みファイルの絶対パスを返す."""
        return settings.weights_dir / weight.relative_path

    def get_all_status(self, weights: list[WeightFileInfo]) -> list[dict[str, Any]]:
        """全重みファイルのステータスを返す."""
        results = []
        for w in weights:
            path = settings.weights_dir / w.relative_path
            results.append({
                "name": w.name,
                "relative_path": w.relative_path,
                "exists": path.exists(),
                "file_size": path.stat().st_size if path.exists() else None,
                "expected_size": w.size_bytes,
                "downloading": w.relative_path in self._active_downloads,
                "download_progress": self._active_downloads.get(w.relative_path, 0.0),
            })
        return results

    async def download_weight(
        self,
        weight: WeightFileInfo,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> Path:
        """
        重みファイルをダウンロード.

        Args:
            weight: ダウンロード対象の重み情報.
            progress_callback: 進捗コールバック.

        Returns:
            保存先パス.

        Raises:
            WeightDownloadError: ダウンロード失敗.
        """
        dest_path = settings.weights_dir / weight.relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 既にダウンロード中ならスキップ
        if weight.relative_path in self._active_downloads:
            raise WeightDownloadError(
                f"'{weight.name}' is already being downloaded.",
                detail="ダウンロードが既に進行中です。",
            )

        self._active_downloads[weight.relative_path] = 0.0
        temp_path = dest_path.with_suffix(".tmp")

        try:
            await self._download_file(
                url=weight.url,
                dest=temp_path,
                expected_size=weight.size_bytes,
                requires_auth=weight.requires_auth,
                progress_callback=progress_callback,
                weight_key=weight.relative_path,
            )

            # SHA256 チェック
            if weight.sha256:
                actual_hash = await asyncio.to_thread(self._compute_sha256, temp_path)
                if actual_hash != weight.sha256:
                    temp_path.unlink(missing_ok=True)
                    raise WeightDownloadError(
                        f"SHA256 mismatch for '{weight.name}'.",
                        detail=f"期待: {weight.sha256}, 実際: {actual_hash}",
                    )

            # リネーム
            temp_path.rename(dest_path)
            logger.info("Downloaded weight: %s -> %s", weight.name, dest_path)
            return dest_path

        except WeightDownloadError:
            raise
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise WeightDownloadError(
                f"Failed to download '{weight.name}': {e}",
                detail=str(e),
            ) from e
        finally:
            self._active_downloads.pop(weight.relative_path, None)

    async def _download_file(
        self,
        url: str,
        dest: Path,
        expected_size: int | None,
        requires_auth: bool,
        progress_callback: Callable[[float, str], None] | None,
        weight_key: str,
    ) -> None:
        """httpx でファイルをストリーミングダウンロード."""
        headers: dict[str, str] = {}
        if requires_auth and settings.hf_token:
            headers["Authorization"] = f"Bearer {settings.hf_token}"

        proxy_config: dict[str, str] = {}
        if settings.http_proxy:
            proxy_config["http://"] = settings.http_proxy
        if settings.https_proxy:
            proxy_config["https://"] = settings.https_proxy

        async with httpx.AsyncClient(
            proxy=proxy_config or None,  # type: ignore[arg-type]
            follow_redirects=True,
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
        ) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0)) or expected_size or 0
                downloaded = 0

                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            progress = downloaded / total
                            self._active_downloads[weight_key] = progress
                            if progress_callback:
                                progress_callback(
                                    progress,
                                    f"ダウンロード中: {downloaded // (1024*1024)} / {total // (1024*1024)} MB",
                                )

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """SHA256 ハッシュを計算."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()


# シングルトン
weight_manager = WeightManager()
