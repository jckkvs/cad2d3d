"""
3D再構築エンジンの抽象基底クラス.

すべてのエンジン（TripoSR, Trellis, Img2CAD 等）はこのインターフェースを実装する.
Strategy パターンにより、エンジンの差し替えを容易にする.

Implements: F-010 (エンジン選択) | 設計: §4.2 プラグインアーキテクチャ
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from app.models.schemas import (
    EngineCapabilities,
    EngineInfo,
    EngineStatus,
    OutputFormat,
    WeightFileInfo,
)

logger = logging.getLogger(__name__)


# ── データクラス ─────────────────────────────────────────


@dataclass
class ProcessedImage:
    """前処理済み画像データ."""

    path: Path
    view_angle: str = "unknown"
    azimuth: float | None = None
    elevation: float | None = None
    scale_info: dict[str, Any] | None = None  # 抽出されたスケール情報
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionParams:
    """3D再構築パラメータ."""

    output_format: OutputFormat = OutputFormat.GLB
    output_dir: Path = Path(".")
    resolution: int = 256
    quality: str = "medium"  # low / medium / high
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """3D再構築結果."""

    success: bool
    output_path: Path | None = None
    output_format: OutputFormat | None = None
    elapsed_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class ProgressCallback(Protocol):
    """進捗コールバックのプロトコル."""

    def __call__(self, progress: float, message: str) -> None: ...


# ── 抽象基底クラス ───────────────────────────────────────


class ReconstructionEngine(ABC):
    """
    2D→3D変換エンジンの抽象インターフェース.

    新しいエンジンを追加する際は:
    1. このクラスを継承
    2. すべての抽象メソッドを実装
    3. @EngineRegistry.register デコレータを付与
    """

    def __init__(self, weights_dir: Path) -> None:
        """
        Args:
            weights_dir: モデル重みファイルの格納ディレクトリ.
        """
        self._weights_dir = weights_dir
        self._model: Any = None

    # ── 識別情報 ─────────────────────────────────────

    @abstractmethod
    def get_name(self) -> str:
        """エンジンの一意な識別子 (例: 'triposr')."""

    @abstractmethod
    def get_display_name(self) -> str:
        """UI表示用の名前 (例: 'TripoSR')."""

    @abstractmethod
    def get_description(self) -> str:
        """エンジンの説明文."""

    @abstractmethod
    def get_version(self) -> str:
        """エンジンのバージョン文字列."""

    # ── 能力・要件 ───────────────────────────────────

    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities:
        """
        エンジンの能力情報を返す.

        - 単一画像対応 / 複数画像対応
        - CAD出力 / メッシュ出力
        - GPU要件
        """

    @abstractmethod
    def get_required_weights(self) -> list[WeightFileInfo]:
        """
        必要なモデル重みファイルの情報を返す.

        各重みファイルの URL、サイズ、SHA256 ハッシュ、配置先パスを含む.
        """

    # ── 状態チェック ─────────────────────────────────

    def check_ready(self) -> EngineStatus:
        """
        エンジンの準備状態を確認.

        デフォルト実装は必要な重みファイルの存在を確認する.
        サブクラスで追加の依存関係チェックをオーバーライド可能.
        """
        for weight in self.get_required_weights():
            weight_path = self._weights_dir / weight.relative_path
            if not weight_path.exists():
                logger.warning(
                    "Weight file missing: %s (expected at %s)",
                    weight.name,
                    weight_path,
                )
                return EngineStatus.WEIGHTS_MISSING
        return EngineStatus.READY

    def get_info(self) -> EngineInfo:
        """エンジン概要情報を生成."""
        readme_path = self._get_readme_path()
        return EngineInfo(
            name=self.get_name(),
            display_name=self.get_display_name(),
            description=self.get_description(),
            version=self.get_version(),
            capabilities=self.get_capabilities(),
            status=self.check_ready(),
            required_weights=self.get_required_weights(),
            readme_path=str(readme_path) if readme_path and readme_path.exists() else None,
        )

    # ── モデルロード ─────────────────────────────────

    @abstractmethod
    async def load_model(self) -> None:
        """
        モデルをメモリにロード.

        重みファイルの読み込み等の初期化処理.
        長時間かかる場合があるため async.
        """

    @abstractmethod
    async def unload_model(self) -> None:
        """
        モデルをメモリから解放.

        GPU メモリの解放等.
        """

    def is_loaded(self) -> bool:
        """モデルがロード済みかどうか."""
        return self._model is not None

    # ── 3D再構築 ─────────────────────────────────────

    @abstractmethod
    async def reconstruct(
        self,
        images: list[ProcessedImage],
        params: ReconstructionParams,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> ReconstructionResult:
        """
        2D画像群から3Dモデルを生成.

        Args:
            images: 前処理済み画像のリスト.
            params: 再構築パラメータ.
            progress_callback: 進捗通知用コールバック (progress: 0.0~1.0, message: str).

        Returns:
            ReconstructionResult: 生成結果.
        """

    # ── ユーティリティ ───────────────────────────────

    def get_output_formats(self) -> list[OutputFormat]:
        """対応出力フォーマットの一覧."""
        return self.get_capabilities().supported_output_formats

    def _get_readme_path(self) -> Path | None:
        """README_MODEL.md のパスを返す."""
        # エンジンのモジュールと同じディレクトリにある README_MODEL.md を探す
        import inspect

        engine_file = Path(inspect.getfile(type(self)))
        readme = engine_file.parent / "README_MODEL.md"
        return readme if readme.exists() else None

    def _report_progress(
        self,
        callback: Callable[[float, str], None] | None,
        progress: float,
        message: str,
    ) -> None:
        """進捗を安全に報告."""
        if callback is not None:
            try:
                callback(min(max(progress, 0.0), 1.0), message)
            except Exception:
                logger.warning("Progress callback raised an exception", exc_info=True)
