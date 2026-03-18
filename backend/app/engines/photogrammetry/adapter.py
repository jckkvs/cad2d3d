"""
openMVG/openMVS フォトグラメトリ エンジンアダプタ.

複数視点画像からStructure from Motion (SfM) + Multi-View Stereo (MVS) で
高精度3D再構築を行う. 伝統的なコンピュータビジョン手法.

openMVG: https://github.com/openMVG/openMVG (SfM: 複数画像→疎点群)
openMVS: https://github.com/cdcseacave/openMVS (MVS: 疎点群→密メッシュ)
Meshroom: https://github.com/alicevision/Meshroom (GUI/パイプライン統合)

Implements: F-051 (フォトグラメトリ)
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import tempfile
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
class PhotogrammetryEngine(ReconstructionEngine):
    """
    openMVG/openMVS フォトグラメトリエンジン.

    複数視点画像から SfM + MVS で3D再構築を行うパイプライン.
    3枚以上の画像が推奨. AIモデル不要でGPU無しでも動作可能.
    """

    def get_name(self) -> str:
        return "photogrammetry"

    def get_display_name(self) -> str:
        return "フォトグラメトリ (openMVG/openMVS)"

    def get_description(self) -> str:
        return (
            "複数視点画像から Structure from Motion + Multi-View Stereo で "
            "高精度3D再構築。3枚以上の画像推奨。AIモデル不要。"
        )

    def get_version(self) -> str:
        return "1.0.0"

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_single_image=False,
            supports_multi_image=True,
            supports_cad_input=False,
            outputs_mesh=True,
            outputs_cad=False,
            outputs_point_cloud=True,
            supported_output_formats=[
                OutputFormat.OBJ, OutputFormat.STL, OutputFormat.PLY,
            ],
            requires_gpu=False,  # CPU のみでも動作
            estimated_vram_gb=None,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        # フォトグラメトリはモデル重みではなくツールのインストールが必要
        return []

    def check_ready(self) -> EngineStatus:
        """openMVG/openMVSがインストールされているか確認."""
        openmvg_ok = self._check_binary("openMVG_main_SfMInit_ImageListing")
        openmvs_ok = self._check_binary("DensifyPointCloud") or self._check_binary("InterfaceOpenMVG2")
        meshroom_ok = self._check_binary("meshroom_batch")

        if openmvg_ok and openmvs_ok:
            return EngineStatus.READY
        if meshroom_ok:
            return EngineStatus.READY
        return EngineStatus.DEPENDENCY_MISSING

    def _check_binary(self, name: str) -> bool:
        """外部バイナリの存在チェック."""
        try:
            subprocess.run([name, "--help"], capture_output=True, timeout=5)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def load_model(self) -> None:
        """フォトグラメトリはモデルロード不要."""
        self._model = True  # マーカー

    async def unload_model(self) -> None:
        self._model = None

    async def reconstruct(
        self,
        images: list[ProcessedImage],
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ReconstructionResult:
        """
        複数視点画像からフォトグラメトリで3D再構築.

        パイプライン:
        1. SfM Init: 画像の一覧とカメラ情報を準備
        2. Feature Detection: SIFT特徴点の検出
        3. Feature Matching: 画像間対応点のマッチング
        4. SfM Reconstruction: 疎点群 + カメラポーズの推定
        5. MVS Densification: 密点群の生成
        6. Mesh Reconstruction: メッシュの再構築
        7. Mesh Refinement + Texturing: メッシュの最適化
        """
        import time
        start_time = time.time()

        if len(images) < 2:
            return ReconstructionResult(
                success=False,
                error_message="フォトグラメトリには2枚以上の画像が必要です (推奨: 3枚以上)。",
            )

        self._report_progress(progress_callback, 0.05, "画像を準備中...")

        # 作業ディレクトリの準備
        work_dir = params.output_dir / "photogrammetry_work"
        images_dir = work_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # 画像をコピー
        from shutil import copy2
        for i, img in enumerate(images):
            dst = images_dir / f"view_{i:03d}{img.path.suffix}"
            copy2(str(img.path), str(dst))

        # Meshroom パイプラインを試行
        meshroom_available = self._check_binary("meshroom_batch")
        if meshroom_available:
            result = await self._run_meshroom(work_dir, images_dir, params, progress_callback)
        else:
            # openMVG + openMVS パイプライン
            result = await self._run_openmvg_openmvs(work_dir, images_dir, params, progress_callback)

        if result.success:
            result.elapsed_seconds = time.time() - start_time

        return result

    async def _run_meshroom(
        self,
        work_dir: Path,
        images_dir: Path,
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None,
    ) -> ReconstructionResult:
        """Meshroom (AliceVision) パイプラインで実行."""
        output_dir = work_dir / "meshroom_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        self._report_progress(progress_callback, 0.1, "Meshroom パイプラインを開始...")

        cmd = [
            "meshroom_batch",
            "--input", str(images_dir),
            "--output", str(output_dir),
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return ReconstructionResult(
                    success=False,
                    error_message=f"Meshroom 実行エラー: {stderr.decode('utf-8', errors='replace')[:500]}",
                )

            # 出力ファイルを探す
            output_file = self._find_output_mesh(output_dir, params.output_format)
            if output_file:
                self._report_progress(progress_callback, 1.0, "完了!")
                return ReconstructionResult(
                    success=True,
                    output_path=output_file,
                    output_format=params.output_format,
                )
            else:
                return ReconstructionResult(
                    success=False,
                    error_message="Meshroom の出力ファイルが見つかりません。",
                )

        except Exception as e:
            return ReconstructionResult(
                success=False,
                error_message=f"Meshroom エラー: {e}",
            )

    async def _run_openmvg_openmvs(
        self,
        work_dir: Path,
        images_dir: Path,
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None,
    ) -> ReconstructionResult:
        """openMVG → openMVS パイプライン."""
        sfm_dir = work_dir / "sfm"
        mvs_dir = work_dir / "mvs"
        sfm_dir.mkdir(parents=True, exist_ok=True)
        mvs_dir.mkdir(parents=True, exist_ok=True)

        steps = [
            (0.1, "SfM: 画像リスト作成", [
                "openMVG_main_SfMInit_ImageListing",
                "-i", str(images_dir),
                "-o", str(sfm_dir / "matches"),
                "-d", "",  # センサーDB (空文字で自動)
            ]),
            (0.2, "SfM: 特徴点検出", [
                "openMVG_main_ComputeFeatures",
                "-i", str(sfm_dir / "matches" / "sfm_data.json"),
                "-o", str(sfm_dir / "matches"),
            ]),
            (0.35, "SfM: 特徴点マッチング", [
                "openMVG_main_ComputeMatches",
                "-i", str(sfm_dir / "matches" / "sfm_data.json"),
                "-o", str(sfm_dir / "matches"),
            ]),
            (0.5, "SfM: 3D再構築", [
                "openMVG_main_SfM",
                "-i", str(sfm_dir / "matches" / "sfm_data.json"),
                "-m", str(sfm_dir / "matches"),
                "-o", str(sfm_dir / "reconstruction"),
            ]),
        ]

        for progress, msg, cmd in steps:
            self._report_progress(progress_callback, progress, msg)
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await process.communicate()
                if process.returncode != 0:
                    return ReconstructionResult(
                        success=False,
                        error_message=f"SfM ステップ失敗: {msg}\n{stderr.decode('utf-8', errors='replace')[:300]}",
                    )
            except FileNotFoundError:
                return ReconstructionResult(
                    success=False,
                    error_message=f"openMVG/openMVS のバイナリが見つかりません: {cmd[0]}。"
                    f"インストールガイドをご確認ください。",
                )

        # openMVS ステップ
        self._report_progress(progress_callback, 0.6, "MVS: 密点群生成中...")

        try:
            # SfM結果をopenMVS形式に変換
            convert_cmd = [
                "openMVG_main_openMVG2openMVS",
                "-i", str(sfm_dir / "reconstruction" / "sfm_data.bin"),
                "-o", str(mvs_dir / "scene.mvs"),
            ]
            process = await asyncio.create_subprocess_exec(
                *convert_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            # 密点群の生成
            self._report_progress(progress_callback, 0.7, "MVS: 点群を密化中...")
            densify_cmd = ["DensifyPointCloud", str(mvs_dir / "scene.mvs")]
            process = await asyncio.create_subprocess_exec(
                *densify_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            # メッシュ再構築
            self._report_progress(progress_callback, 0.85, "MVS: メッシュ再構築中...")
            mesh_cmd = ["ReconstructMesh", str(mvs_dir / "scene_dense.mvs")]
            process = await asyncio.create_subprocess_exec(
                *mesh_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await process.communicate()

            # 結果ファイルを探す
            output_file = self._find_output_mesh(mvs_dir, params.output_format)
            if output_file:
                self._report_progress(progress_callback, 1.0, "完了!")
                return ReconstructionResult(
                    success=True,
                    output_path=output_file,
                    output_format=params.output_format,
                )

        except FileNotFoundError:
            pass

        return ReconstructionResult(
            success=False,
            error_message="openMVS のバイナリが見つかりません。",
        )

    def _find_output_mesh(self, directory: Path, fmt: OutputFormat | None) -> Path | None:
        """出力ディレクトリからメッシュファイルを検索."""
        search_exts = [".obj", ".ply", ".stl", ".glb"]
        if fmt:
            search_exts.insert(0, f".{fmt.value}")

        for ext in search_exts:
            for f in directory.rglob(f"*{ext}"):
                return f
        return None
