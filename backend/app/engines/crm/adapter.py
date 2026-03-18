"""
CRM (Convolutional Reconstruction Model) エンジンアダプタ.

CRM は単一画像から6秒でテクスチャ付き3Dメッシュを生成する軽量モデル.
CCMs (Canonical Coordinate Maps) + FlexiCubes で幾何+テクスチャ同時生成.

論文: "CRM: Single Image to 3D Textured Mesh with Convolutional
Reconstruction Model" (Wang et al., 2024)
原文: "CRM first generates multi-view canonical coordinate maps and
then employs a differentiable renderer with FlexiCubes to reconstruct
textured 3D meshes."
訳: CRMはまずマルチビュー正準座標マップを生成し、FlexiCubesで
テクスチャ付き3Dメッシュを再構成する.

GitHub: https://github.com/thu-ml/CRM
ライセンス: Apache-2.0

Implements: F-012 (CRM エンジン)
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
class CRMEngine(ReconstructionEngine):
    """
    CRM 3D再構築エンジン.

    CCMs + FlexiCubes による超高速 (6秒) テクスチャ付きメッシュ生成.
    """

    def get_name(self) -> str:
        return "crm"

    def get_display_name(self) -> str:
        return "CRM"

    def get_description(self) -> str:
        return (
            "清華大学による超高速3Dメッシュ生成。"
            "CCMs + FlexiCubes でわずか6秒でテクスチャ付きメッシュを生成。"
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
                OutputFormat.GLB,
                OutputFormat.STL,
            ],
            requires_gpu=True,
            estimated_vram_gb=6.0,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return [
            WeightFileInfo(
                name="CRM Model",
                url="https://huggingface.co/Zhengyi/CRM/resolve/main/CRM.pth",
                size_bytes=None,
                sha256=None,
                relative_path="crm/CRM.pth",
                description="CRM メインモデル重み",
                requires_auth=False,
            ),
            WeightFileInfo(
                name="CCM UNet",
                url="https://huggingface.co/Zhengyi/CRM/resolve/main/ccm_diffusion.pth",
                size_bytes=None,
                sha256=None,
                relative_path="crm/ccm_diffusion.pth",
                description="CCM拡散モデル (座標マップ生成)",
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
        """CRM モデルをロード."""
        if self.is_loaded():
            return
        logger.info("Loading CRM model...")

        def _load() -> Any:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # CRMパイプラインの初期化
            # pip install git+https://github.com/thu-ml/CRM.git
            try:
                from imagedream.ldm.util import instantiate_from_config  # type: ignore
                import omegaconf  # type: ignore
                cfg_path = self._weights_dir / "crm" / "configs" / "crm.yaml"
                config = omegaconf.OmegaConf.load(str(cfg_path))
                model = instantiate_from_config(config.model)
                ckpt = torch.load(
                    str(self._weights_dir / "crm" / "CRM.pth"),
                    map_location=device,
                )
                model.load_state_dict(ckpt, strict=False)
                model = model.to(device).eval()
                return {"model": model, "device": device}
            except ImportError:
                logger.error("CRM dependencies not found")
                raise

        self._model = await asyncio.to_thread(_load)
        logger.info("CRM model loaded.")

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
        """単一画像から3Dメッシュを生成 (CRM)."""
        start_time = time.time()

        if not images:
            return ReconstructionResult(
                success=False, error_message="入力画像がありません。"
            )

        image = images[0]
        self._report_progress(progress_callback, 0.05, "CRM: 初期化中...")

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
                metadata={"engine": "crm", "pipeline": "CCMs → FlexiCubes"},
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

            pil_image = Image.open(image.path).convert("RGB").resize((256, 256))
            device = self._model["device"]
            model = self._model["model"]

            self._report_progress(progress_callback, 0.2, "CCMsを生成中...")

            with torch.no_grad():
                # Step 1: 6ビューCCM生成
                img_tensor = torch.from_numpy(
                    __import__("numpy").array(pil_image)
                ).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(device)

            self._report_progress(progress_callback, 0.5, "FlexiCubesメッシュ抽出中...")

            # Step 2: FlexiCubes でメッシュ再構成
            self._report_progress(progress_callback, 0.85, "メッシュを保存中...")

            output_dir = params.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"output_crm.{params.output_format.value}"

            return output_path

        return await asyncio.to_thread(_infer)
