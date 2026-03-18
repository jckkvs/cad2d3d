"""
TripoSR エンジンアダプタ.

TripoSR は Stability AI と Tripo AI による単一画像→3Dメッシュ生成モデル.
MIT ライセンス、1秒以下で高速生成可能.

論文: "TripoSR: Fast 3D Object Reconstruction from a Single Image" (2024)
GitHub: https://github.com/VAST-AI-Research/TripoSR
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

from app.engines.base import (
    ProcessedImage,
    ReconstructionEngine,
    ReconstructionParams,
    ReconstructionResult,
)
from app.engines.registry import EngineRegistry
from app.models.schemas import (
    EngineCapabilities,
    EngineStatus,
    OutputFormat,
    WeightFileInfo,
)

logger = logging.getLogger(__name__)


@EngineRegistry.register
class TripoSREngine(ReconstructionEngine):
    """
    TripoSR 3D再構築エンジン.

    単一画像から高品質3Dメッシュを高速生成する.
    MITライセンスで商用利用可能.
    """

    def get_name(self) -> str:
        return "triposr"

    def get_display_name(self) -> str:
        return "TripoSR"

    def get_description(self) -> str:
        return (
            "Stability AI による単一画像→3Dメッシュ高速生成モデル。"
            "1秒以下で高品質な3Dメッシュを生成。MIT ライセンス。"
        )

    def get_version(self) -> str:
        return "1.0.0"

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_single_image=True,
            supports_multi_image=False,
            supports_cad_input=False,
            outputs_mesh=True,
            outputs_cad=False,
            outputs_point_cloud=False,
            supported_output_formats=[
                OutputFormat.OBJ,
                OutputFormat.STL,
                OutputFormat.GLB,
                OutputFormat.GLTF,
                OutputFormat.PLY,
            ],
            requires_gpu=True,
            estimated_vram_gb=8.0,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return [
            WeightFileInfo(
                name="TripoSR Model",
                url="https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt",
                size_bytes=None,  # サイズは HF 上で確認
                sha256=None,
                relative_path="triposr/model.ckpt",
                description="TripoSR メインモデルの重みファイル",
                requires_auth=False,
            ),
        ]

    def check_ready(self) -> EngineStatus:
        """依存関係チェック: torch がインストールされているか."""
        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("PyTorch is not installed. TripoSR requires GPU support.")
            return EngineStatus.DEPENDENCY_MISSING

        return super().check_ready()

    async def load_model(self) -> None:
        """TripoSR モデルをロード."""
        if self.is_loaded():
            logger.info("TripoSR model already loaded.")
            return

        logger.info("Loading TripoSR model...")

        def _load() -> Any:
            try:
                # TripoSR の tsr モジュールを動的インポート
                # pip install triposr or from source
                from tsr.system import TSR  # type: ignore[import-not-found]

                model = TSR.from_pretrained(
                    "stabilityai/TripoSR",
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                model.renderer.set_chunk_size(8192)
                model.to("cuda" if self._is_cuda_available() else "cpu")
                return model
            except ImportError:
                logger.error(
                    "TripoSR package not found. Install: pip install triposr"
                )
                raise
            except Exception:
                logger.error("Failed to load TripoSR model", exc_info=True)
                raise

        self._model = await asyncio.to_thread(_load)
        logger.info("TripoSR model loaded successfully.")

    async def unload_model(self) -> None:
        """TripoSR モデルをアンロード."""
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("TripoSR model unloaded.")

    async def reconstruct(
        self,
        images: list[ProcessedImage],
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ReconstructionResult:
        """
        単一画像から3Dメッシュを生成.

        Args:
            images: 1枚の前処理済み画像 (TripoSR は単一画像のみ対応).
            params: 再構築パラメータ.
            progress_callback: 進捗コールバック.

        Returns:
            ReconstructionResult
        """
        start_time = time.time()

        if not images:
            return ReconstructionResult(
                success=False,
                error_message="入力画像がありません。",
            )

        if len(images) > 1:
            logger.warning(
                "TripoSR supports single image only. Using first image."
            )

        image = images[0]
        self._report_progress(progress_callback, 0.05, "モデルをロード中...")

        # モデルがロードされていなければロード
        if not self.is_loaded():
            await self.load_model()

        self._report_progress(progress_callback, 0.15, "画像を前処理中...")

        try:
            result_path = await self._run_inference(image, params, progress_callback)

            elapsed = time.time() - start_time
            self._report_progress(progress_callback, 1.0, "完了!")

            return ReconstructionResult(
                success=True,
                output_path=result_path,
                output_format=params.output_format,
                elapsed_seconds=elapsed,
                metadata={
                    "engine": self.get_name(),
                    "resolution": params.resolution,
                    "input_image": str(image.path),
                },
            )
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("TripoSR reconstruction failed", exc_info=True)
            return ReconstructionResult(
                success=False,
                elapsed_seconds=elapsed,
                error_message=str(e),
            )

    async def _run_inference(
        self,
        image: ProcessedImage,
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None,
    ) -> Path:
        """推論をバックグラウンドスレッドで実行."""

        def _infer() -> Path:
            import numpy as np
            from PIL import Image

            # 画像読み込み
            pil_image = Image.open(image.path).convert("RGB")

            self._report_progress(progress_callback, 0.3, "3Dモデルを生成中...")

            # TripoSR 推論
            with __import__("torch").no_grad():
                scene_codes = self._model(pil_image, device="cuda" if self._is_cuda_available() else "cpu")

            self._report_progress(progress_callback, 0.7, "メッシュを抽出中...")

            # メッシュ抽出
            meshes = self._model.extract_mesh(
                scene_codes,
                resolution=params.resolution,
            )
            mesh = meshes[0]

            self._report_progress(progress_callback, 0.85, "ファイルを保存中...")

            # 出力パスの決定
            output_dir = params.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            output_ext = params.output_format.value
            output_path = output_dir / f"output.{output_ext}"

            # trimesh で保存
            import trimesh

            tri_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
            )
            tri_mesh.export(str(output_path))

            return output_path

        return await asyncio.to_thread(_infer)

    @staticmethod
    def _is_cuda_available() -> bool:
        """CUDA が利用可能かチェック."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False
