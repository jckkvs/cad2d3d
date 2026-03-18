"""
Hunyuan3D-2 エンジンアダプタ.

Hunyuan3D 2.0 は Tencent Hunyuan の画像→3D生成モデル.
高品質テクスチャ付きメッシュを生成可能.

論文: "Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution
Textured 3D Assets Generation"
URL: https://arxiv.org/abs/2501.12202
GitHub: https://github.com/Tencent/Hunyuan3D-2

Implements: F-052 (Hunyuan3D-2)
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Callable

from app.engines.base import (
    ReconstructionEngine,
    ProcessedImage,
    ReconstructionParams,
    ReconstructionResult,
)
from app.engines.registry import EngineRegistry
from app.models.schemas import (
    EngineCapabilities,
    OutputFormat,
    WeightFileInfo,
)

logger = logging.getLogger(__name__)


@EngineRegistry.register
class Hunyuan3D2Engine(ReconstructionEngine):
    """
    Hunyuan3D 2.0 エンジン.

    論文原文引用:
    "We present Hunyuan3D 2.0, a novel two-stage 3D asset generation
    framework that produces high-resolution textured 3D assets."
    日本語訳: 高解像度テクスチャ付き3Dアセットを生成する、
    2段階の新しい3Dアセット生成フレームワーク Hunyuan3D 2.0 を提案する。
    """

    def get_name(self) -> str:
        return "hunyuan3d2"

    def get_display_name(self) -> str:
        return "Hunyuan3D 2.0 (Tencent)"

    def get_description(self) -> str:
        return (
            "Tencent の高解像度テクスチャ付き3Dアセット生成モデル。"
            "2段階パイプライン: 形状生成 → テクスチャ生成。"
            "PBRマテリアル対応。"
        )

    def get_version(self) -> str:
        return "2.0.0"

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_single_image=True,
            supports_multi_image=True,
            supports_cad_input=False,
            outputs_mesh=True,
            outputs_cad=False,
            outputs_point_cloud=False,
            supported_output_formats=[
                OutputFormat.GLB, OutputFormat.OBJ, OutputFormat.STL,
            ],
            requires_gpu=True,
            estimated_vram_gb=16.0,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return [
            WeightFileInfo(
                name="Hunyuan3D-2 Shape Model",
                url="https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-dit-v2-0/model.safetensors",
                relative_path="hunyuan3d2/shape/model.safetensors",
                description="形状生成 DiT モデル重み",
                requires_auth=False,
            ),
            WeightFileInfo(
                name="Hunyuan3D-2 Texture Model",
                url="https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-paint-v2-0/model.safetensors",
                relative_path="hunyuan3d2/texture/model.safetensors",
                description="テクスチャ生成モデル重み",
                requires_auth=False,
            ),
        ]

    async def load_model(self) -> None:
        """Hunyuan3D-2 モデルをロード."""
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import]
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # type: ignore[import]

            weights_dir = self._weights_dir / "hunyuan3d2"

            shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                "tencent/Hunyuan3D-2",
                subfolder="hunyuan3d-dit-v2-0",
                cache_dir=str(weights_dir),
            )

            texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                "tencent/Hunyuan3D-2",
                subfolder="hunyuan3d-paint-v2-0",
                cache_dir=str(weights_dir),
            )

            self._model = {
                "shape": shape_pipeline,
                "texture": texture_pipeline,
            }
            logger.info("Hunyuan3D-2 models loaded")
        except ImportError:
            logger.warning("hy3dgen パッケージがインストールされていません")
            self._model = None
        except Exception as e:
            logger.error("Hunyuan3D-2 load failed: %s", e)
            self._model = None

    async def unload_model(self) -> None:
        self._model = None

    async def reconstruct(
        self,
        images: list[ProcessedImage],
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ReconstructionResult:
        """Hunyuan3D-2 で3Dモデルを生成."""
        start = time.time()

        if not self._model:
            await self.load_model()
            if not self._model:
                return ReconstructionResult(
                    success=False,
                    error_message="Hunyuan3D-2 のロードに失敗しました。hy3dgen パッケージをインストールしてください。",
                )

        try:
            from PIL import Image

            img = Image.open(images[0].path).convert("RGB")
            self._report_progress(progress_callback, 0.1, "形状を生成中...")

            # Stage 1: 形状生成
            shape_result = await asyncio.to_thread(
                self._model["shape"],
                image=img,
                num_inference_steps=50,
            )

            self._report_progress(progress_callback, 0.5, "テクスチャを生成中...")

            # Stage 2: テクスチャ生成
            textured = await asyncio.to_thread(
                self._model["texture"],
                mesh=shape_result,
                image=img,
            )

            self._report_progress(progress_callback, 0.9, "ファイルに保存中...")

            output_format = params.output_format or OutputFormat.GLB
            output_path = params.output_dir / f"hunyuan3d2_output.{output_format.value}"

            if hasattr(textured, "export"):
                textured.export(str(output_path))
            else:
                import trimesh
                mesh = trimesh.Trimesh(
                    vertices=textured.vertices,
                    faces=textured.faces,
                )
                mesh.export(str(output_path))

            self._report_progress(progress_callback, 1.0, "完了!")

            return ReconstructionResult(
                success=True,
                output_path=output_path,
                output_format=output_format,
                elapsed_seconds=time.time() - start,
                metadata={"model": "Hunyuan3D-2.0"},
            )

        except Exception as e:
            logger.error("Hunyuan3D-2 reconstruction failed", exc_info=True)
            return ReconstructionResult(
                success=False,
                error_message=f"Hunyuan3D-2 生成エラー: {e}",
            )
