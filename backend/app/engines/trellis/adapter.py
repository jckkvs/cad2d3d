"""
Trellis エンジンアダプタ.

Trellis は Microsoft Research の高品質3DアセットN生成モデル.
Structured 3D Latent (SLAT) 表現を使用し、メッシュ・ガウシアン等を出力.

論文: "Structured 3D Latents for Scalable and Versatile 3D Generation"
原文 URL: https://arxiv.org/abs/2412.01506
GitHubリポジトリ: https://github.com/microsoft/TRELLIS

Implements: F-050 (Trellis生成エンジン)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from app.engines.base import (
    ReconstructionEngine,
    ProcessedImage,
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
class TrellisEngine(ReconstructionEngine):
    """
    Trellis 3Dアセット生成エンジン.

    Structured 3D Latent (SLAT) 表現を使用し、
    テキストまたは画像入力から高品質な3Dアセットを生成.

    論文: "Structured 3D Latents for Scalable and Versatile 3D Generation"
    原文引用: "We introduce a novel 3D generation method based on
    Structured 3D Latents (SLAT), a flexible and powerful representation
    that encodes 3D objects into a set of sparse latent tokens."
    日本語訳: 我々は Structured 3D Latents (SLAT) に基づく新しい3D生成手法を提案する。
    SLATは疎な潜在トークンの集合に3Dオブジェクトをエンコードする
    柔軟かつ強力な表現である。
    """

    def get_name(self) -> str:
        return "trellis"

    def get_display_name(self) -> str:
        return "Trellis (Microsoft)"

    def get_description(self) -> str:
        return (
            "Microsoft Research の高品質3Dアセット生成モデル。"
            "Structured 3D Latent (SLAT) 表現により、"
            "テクスチャ付き高品質メッシュを生成。"
        )

    def get_version(self) -> str:
        return "1.0.0"

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_single_image=True,
            supports_multi_image=True,
            supports_cad_input=False,
            outputs_mesh=True,
            outputs_cad=False,
            outputs_point_cloud=True,
            supported_output_formats=[
                OutputFormat.GLB, OutputFormat.OBJ, OutputFormat.STL, OutputFormat.PLY,
            ],
            requires_gpu=True,
            estimated_vram_gb=12.0,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return [
            WeightFileInfo(
                name="Trellis SLAT Model",
                url="https://huggingface.co/microsoft/TRELLIS-image-large/resolve/main/pipeline.json",
                relative_path="trellis/pipeline.json",
                description="Trellis パイプライン設定ファイル",
                requires_auth=False,
            ),
        ]

    async def load_model(self) -> None:
        """Trellis モデルをロード."""
        try:
            # trellis パッケージがインストールされているか確認
            from trellis.pipelines import TrellisImageTo3DPipeline  # type: ignore[import]

            weights_path = self._weights_dir / "trellis"
            self._model = TrellisImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-image-large",
                cache_dir=str(weights_path),
            )
            self._model.to("cuda")
            logger.info("Trellis model loaded successfully")
        except ImportError:
            logger.warning("trellis パッケージがインストールされていません")
            self._model = None
        except Exception as e:
            logger.error("Trellis model load failed: %s", e)
            self._model = None

    async def unload_model(self) -> None:
        """モデルをアンロード."""
        self._model = None

    async def reconstruct(
        self,
        images: list[ProcessedImage],
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ReconstructionResult:
        """
        Trellis で3Dモデルを生成.

        Implements: Trellis の推論パイプライン
        """
        import time
        start = time.time()

        self._report_progress(progress_callback, 0.1, "画像を読み込み中...")

        if not self._model:
            await self.load_model()
            if not self._model:
                return ReconstructionResult(
                    success=False,
                    error_message="Trellis モデルのロードに失敗しました。trellis パッケージをインストールしてください。",
                )

        try:
            from PIL import Image

            # 複数画像の場合は最初の画像を使用
            img = Image.open(images[0].path).convert("RGB")

            self._report_progress(progress_callback, 0.3, "SLAT潜在空間にエンコード中...")

            # 推論実行 (非同期でラップ)
            outputs = await asyncio.to_thread(
                self._model.run, img,
                seed=42,
            )

            self._report_progress(progress_callback, 0.7, "3Dメッシュをデコード中...")

            # メッシュ抽出
            output_format = params.output_format or OutputFormat.GLB
            ext = output_format.value
            output_path = params.output_dir / f"trellis_output.{ext}"

            mesh = outputs['mesh'][0] if isinstance(outputs.get('mesh'), list) else outputs.get('mesh')

            if mesh is not None:
                # trimesh で保存
                import trimesh
                if hasattr(mesh, 'export'):
                    mesh.export(str(output_path))
                else:
                    # vertices + faces の場合
                    t_mesh = trimesh.Trimesh(
                        vertices=mesh.vertices if hasattr(mesh, 'vertices') else mesh[0],
                        faces=mesh.faces if hasattr(mesh, 'faces') else mesh[1],
                    )
                    t_mesh.export(str(output_path))

            self._report_progress(progress_callback, 1.0, "完了!")

            elapsed = time.time() - start
            return ReconstructionResult(
                success=True,
                output_path=output_path,
                output_format=output_format,
                elapsed_seconds=elapsed,
                metadata={"model": "TRELLIS-image-large", "seed": 42},
            )

        except Exception as e:
            logger.error("Trellis reconstruction failed", exc_info=True)
            return ReconstructionResult(
                success=False,
                error_message=f"Trellis 生成エラー: {e}",
            )
