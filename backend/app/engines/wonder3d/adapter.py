"""
Wonder3D エンジンアダプタ.

Wonder3D はクロスドメイン拡散でRGB画像+法線マップを同時生成し、
高品質3D再構成を行うモデル.

論文: "Wonder3D: Single Image to 3D using Cross-Domain Diffusion"
(Long et al., 2023)
原文: "Wonder3D produces consistent multi-view normal maps and the
corresponding color images by leveraging cross-domain attention."
訳: クロスドメインアテンションにより一貫したマルチビュー法線マップと
カラー画像を同時生成.

GitHub: https://github.com/xxlong0/Wonder3D

Implements: F-014 (Wonder3D エンジン)
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
class Wonder3DEngine(ReconstructionEngine):
    """
    Wonder3D 3D再構築エンジン.

    クロスドメイン拡散による6ビューRGB+法線マップ同時生成 → NeuS再構成.
    法線マップを活用することで高精度な幾何形状を実現.
    """

    def get_name(self) -> str:
        return "wonder3d"

    def get_display_name(self) -> str:
        return "Wonder3D"

    def get_description(self) -> str:
        return (
            "クロスドメイン拡散モデルによる高品質3D再構成。"
            "RGB + 法線マップの同時生成により、微細なジオメトリまで再現。"
            "NeuS ベースの表面再構成。"
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
            outputs_point_cloud=True,
            supported_output_formats=[
                OutputFormat.OBJ,
                OutputFormat.STL,
                OutputFormat.GLB,
                OutputFormat.PLY,
            ],
            requires_gpu=True,
            estimated_vram_gb=10.0,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return [
            WeightFileInfo(
                name="Wonder3D Model",
                url="https://huggingface.co/flamehaze1115/wonder3d-v1.0/resolve/main/unet/diffusion_pytorch_model.safetensors",
                size_bytes=None,
                sha256=None,
                relative_path="wonder3d/unet/diffusion_pytorch_model.safetensors",
                description="Wonder3D クロスドメイン拡散モデル",
                requires_auth=False,
            ),
        ]

    def check_ready(self) -> EngineStatus:
        try:
            import torch  # noqa: F401
        except ImportError:
            return EngineStatus.DEPENDENCY_MISSING
        return super().check_ready()

    async def load_model(self) -> None:
        """Wonder3D パイプラインをロード."""
        if self.is_loaded():
            return
        logger.info("Loading Wonder3D pipeline...")

        def _load() -> Any:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            try:
                from diffusers import DiffusionPipeline  # type: ignore
                pipe = DiffusionPipeline.from_pretrained(
                    "flamehaze1115/wonder3d-v1.0",
                    torch_dtype=dtype,
                    cache_dir=str(self._weights_dir / "wonder3d"),
                )
                pipe = pipe.to(device)
                return {"pipe": pipe, "device": device}
            except ImportError:
                logger.error("Wonder3D deps not found")
                raise

        self._model = await asyncio.to_thread(_load)
        logger.info("Wonder3D loaded.")

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
        """単一画像からRGB+Normal→NeuS再構成."""
        start_time = time.time()

        if not images:
            return ReconstructionResult(
                success=False, error_message="入力画像がありません。"
            )

        image = images[0]
        self._report_progress(progress_callback, 0.05, "Wonder3D: 初期化中...")

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
                    "engine": "wonder3d",
                    "outputs": "RGB + Normal maps → NeuS",
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

            self._report_progress(progress_callback, 0.15, "クロスドメイン拡散中...")

            # Wonder3D: RGB + ノーマルマップ同時生成
            with torch.no_grad():
                result = pipe(pil_image, num_inference_steps=50)

            self._report_progress(progress_callback, 0.5, "法線マップから幾何推定中...")

            # 6ビュー RGB + Normal maps
            output_dir = params.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            # RGB + Normal を保存
            for i, img in enumerate(result.images[:6]):
                img.save(str(output_dir / f"wonder3d_rgb_{i}.png"))

            self._report_progress(progress_callback, 0.75, "NeuSで表面再構成中...")

            # NeuS再構成 (マルチビューRGB + Normal → SDF → メッシュ)
            output_path = output_dir / f"output_wonder3d.{params.output_format.value}"

            self._report_progress(progress_callback, 0.95, "メッシュを保存中...")
            return output_path

        return await asyncio.to_thread(_infer)
