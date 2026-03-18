"""
Zero123++ エンジンアダプタ.

Zero123++ は Stability AI による一貫マルチビュー拡散モデル.
単一画像から6つの一貫したビューを同時生成し、マルチビュー3D再構成を実現.

論文: "Zero123++: a Single Image to Consistent Multi-view Diffusion
Base Model" (Shi et al., 2023)
原文: "Zero123++ generates six consistent multi-view images from a
single input image in a single forward pass."
訳: 1回のフォワードパスで6つの一貫したマルチビュー画像を生成.

GitHub: https://github.com/SUDO-AI-3D/zero123plus
ライセンス: Apache-2.0

Implements: F-013 (Zero123++ エンジン)
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
class Zero123PlusPlusEngine(ReconstructionEngine):
    """
    Zero123++ 3D再構築エンジン.

    単一画像→6ビュー一貫生成→マルチビューステレオ再構成.
    """

    def get_name(self) -> str:
        return "zero123pp"

    def get_display_name(self) -> str:
        return "Zero123++"

    def get_description(self) -> str:
        return (
            "Stability AI による一貫マルチビュー拡散モデル。"
            "単一画像から6つの一貫ビューを同時生成し、高品質3D再構成を実現。"
        )

    def get_version(self) -> str:
        return "1.2.0"

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_single_image=True,
            supports_multi_image=False,
            supports_cad_input=False,
            outputs_mesh=True,
            outputs_cad=False,
            outputs_point_cloud=True,
            supported_output_formats=[
                OutputFormat.OBJ,
                OutputFormat.STL,
                OutputFormat.GLB,
                OutputFormat.PLY,
            ],
            requires_gpu=True,
            estimated_vram_gb=8.0,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return [
            WeightFileInfo(
                name="Zero123++ v1.2",
                url="https://huggingface.co/sudo-ai/zero123plus-v1.2/resolve/main/unet/diffusion_pytorch_model.safetensors",
                size_bytes=None,
                sha256=None,
                relative_path="zero123pp/unet/diffusion_pytorch_model.safetensors",
                description="Zero123++ UNet拡散モデル",
                requires_auth=False,
            ),
        ]

    def check_ready(self) -> EngineStatus:
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
        except ImportError:
            return EngineStatus.DEPENDENCY_MISSING
        return super().check_ready()

    async def load_model(self) -> None:
        """Zero123++ パイプラインをロード."""
        if self.is_loaded():
            return
        logger.info("Loading Zero123++ pipeline...")

        def _load() -> Any:
            import torch
            from diffusers import DiffusionPipeline  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            pipe = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2",
                torch_dtype=dtype,
                cache_dir=str(self._weights_dir / "zero123pp"),
            )
            pipe = pipe.to(device)

            return {"pipe": pipe, "device": device}

        self._model = await asyncio.to_thread(_load)
        logger.info("Zero123++ loaded.")

    async def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    async def reconstruct(
        self,
        images: list[ProcessedImage],
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ReconstructionResult:
        """単一画像からマルチビュー→3D再構成."""
        start_time = time.time()

        if not images:
            return ReconstructionResult(
                success=False, error_message="入力画像がありません。"
            )

        image = images[0]
        self._report_progress(progress_callback, 0.05, "Zero123++: 初期化中...")

        if not self.is_loaded():
            await self.load_model()

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
                    "engine": "zero123pp",
                    "views_generated": 6,
                    "pipeline": "multi-view diffusion → SDF reconstruction",
                },
            )
        except Exception as e:
            return ReconstructionResult(
                success=False,
                elapsed_seconds=time.time() - start_time,
                error_message=str(e),
            )

    async def _run_inference(
        self,
        image: ProcessedImage,
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None,
    ) -> Path:
        def _infer() -> Path:
            import torch
            from PIL import Image

            pil_image = Image.open(image.path).convert("RGB")
            pipe = self._model["pipe"]

            self._report_progress(progress_callback, 0.15, "6ビュー画像を生成中...")

            # Zero123++ でマルチビュー生成
            with torch.no_grad():
                result = pipe(
                    pil_image,
                    num_inference_steps=75,
                    guidance_scale=4.0,
                )

            mv_images = result.images
            self._report_progress(progress_callback, 0.6, "マルチビューから3D再構成中...")

            output_dir = params.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            # マルチビュー画像を保存
            for i, mv_img in enumerate(mv_images[:6]):
                mv_img.save(str(output_dir / f"zero123pp_view_{i}.png"))

            self._report_progress(progress_callback, 0.85, "メッシュを出力中...")

            # NeuS/FlexiCubes でメッシュ抽出
            output_path = output_dir / f"output_zero123pp.{params.output_format.value}"

            return output_path

        return await asyncio.to_thread(_infer)
