"""
EngineRegistry と ReconstructionEngine のユニットテスト.

テスト対象:
- エンジンの登録 / 自動検出
- エンジンの取得 / 一覧
- 準備状態チェック
- 名前衝突時の挙動
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

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
from app.core.exceptions import EngineNotFoundError


# ── テスト用ダミーエンジン ──────────────────────

class DummyEngine(ReconstructionEngine):
    """テスト用の最小限エンジン実装."""

    def get_name(self) -> str:
        return "dummy"

    def get_display_name(self) -> str:
        return "Dummy Engine"

    def get_description(self) -> str:
        return "テスト用ダミーエンジン"

    def get_version(self) -> str:
        return "0.0.1"

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_single_image=True,
            supports_multi_image=False,
            outputs_mesh=True,
            outputs_cad=False,
            supported_output_formats=[OutputFormat.OBJ, OutputFormat.STL],
            requires_gpu=False,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return []

    async def load_model(self) -> None:
        self._model = "loaded"

    async def unload_model(self) -> None:
        self._model = None

    async def reconstruct(
        self, images, params, progress_callback=None,
    ) -> ReconstructionResult:
        if progress_callback:
            progress_callback(0.5, "halfway")
            progress_callback(1.0, "done")
        return ReconstructionResult(
            success=True,
            output_path=params.output_dir / "test.obj",
            output_format=OutputFormat.OBJ,
            elapsed_seconds=0.1,
        )


class DummyEngineWithWeights(ReconstructionEngine):
    """重みファイルが必要なテスト用エンジン."""

    def get_name(self) -> str:
        return "dummy_weights"

    def get_display_name(self) -> str:
        return "Dummy (with weights)"

    def get_description(self) -> str:
        return "重みファイルが必要なテストエンジン"

    def get_version(self) -> str:
        return "0.0.1"

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_single_image=True,
            outputs_mesh=True,
            supported_output_formats=[OutputFormat.GLB],
            requires_gpu=True,
            estimated_vram_gb=4.0,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return [
            WeightFileInfo(
                name="test_model.bin",
                url="https://example.com/model.bin",
                relative_path="dummy_weights/model.bin",
                description="Dummy weight file",
            ),
        ]

    async def load_model(self) -> None:
        self._model = "loaded"

    async def unload_model(self) -> None:
        self._model = None

    async def reconstruct(self, images, params, progress_callback=None) -> ReconstructionResult:
        return ReconstructionResult(success=True)


# ── フィクスチャ ──────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    """各テスト前にレジストリをリセット."""
    EngineRegistry.reset()
    yield
    EngineRegistry.reset()


@pytest.fixture
def tmp_weights_dir(tmp_path: Path) -> Path:
    """テスト用の一時weights ディレクトリ."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    return weights_dir


# ── テスト: エンジン登録 ─────────────────────

class TestEngineRegistry:
    """EngineRegistry のテスト."""

    def test_register_engine(self, tmp_weights_dir: Path) -> None:
        """エンジンを正常に登録できる."""
        initial_count = len(EngineRegistry.list_available())
        EngineRegistry.register(DummyEngine)
        engines = EngineRegistry.list_available()
        assert len(engines) == initial_count + 1
        names = {e.name for e in engines}
        assert "dummy" in names

    def test_register_multiple_engines(self, tmp_weights_dir: Path) -> None:
        """複数エンジンを登録できる."""
        EngineRegistry.register(DummyEngine)
        EngineRegistry.register(DummyEngineWithWeights)
        engines = EngineRegistry.list_available()
        assert len(engines) == 2
        names = {e.name for e in engines}
        assert names == {"dummy", "dummy_weights"}

    def test_get_engine(self, tmp_weights_dir: Path) -> None:
        """名前でエンジンインスタンスを取得できる."""
        EngineRegistry.register(DummyEngine)
        engine = EngineRegistry.get("dummy")
        assert engine.get_name() == "dummy"
        assert isinstance(engine, DummyEngine)

    def test_get_same_instance(self, tmp_weights_dir: Path) -> None:
        """同じ名前で取得するとき同一インスタンスを返す."""
        EngineRegistry.register(DummyEngine)
        e1 = EngineRegistry.get("dummy")
        e2 = EngineRegistry.get("dummy")
        assert e1 is e2

    def test_get_nonexistent_engine(self) -> None:
        """存在しないエンジンの取得で EngineNotFoundError."""
        with pytest.raises(EngineNotFoundError):
            EngineRegistry.get("nonexistent")

    def test_reset(self, tmp_weights_dir: Path) -> None:
        """reset() でレジストリがクリアされる."""
        EngineRegistry.register(DummyEngine)
        assert len(EngineRegistry.list_available()) == 1
        EngineRegistry.reset()
        assert len(EngineRegistry.list_available()) == 0


# ── テスト: エンジン基底クラス ────────────────

class TestReconstructionEngine:
    """ReconstructionEngine の基本動作テスト."""

    def test_engine_info(self, tmp_weights_dir: Path) -> None:
        """get_info() が正しいEngineInfoを返す."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        info = engine.get_info()
        assert info.name == "dummy"
        assert info.display_name == "Dummy Engine"
        assert info.capabilities.supports_single_image is True
        assert info.capabilities.requires_gpu is False

    def test_check_ready_no_weights(self, tmp_weights_dir: Path) -> None:
        """重みが不要なエンジンは READY."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        assert engine.check_ready() == EngineStatus.READY

    def test_check_ready_missing_weights(self, tmp_weights_dir: Path) -> None:
        """重みファイルが不足している場合 WEIGHTS_MISSING."""
        engine = DummyEngineWithWeights(weights_dir=tmp_weights_dir)
        assert engine.check_ready() == EngineStatus.WEIGHTS_MISSING

    def test_check_ready_weights_exist(self, tmp_weights_dir: Path) -> None:
        """重みファイルが存在する場合 READY."""
        engine = DummyEngineWithWeights(weights_dir=tmp_weights_dir)
        weight_path = tmp_weights_dir / "dummy_weights" / "model.bin"
        weight_path.parent.mkdir(parents=True)
        weight_path.write_text("fake model data")
        assert engine.check_ready() == EngineStatus.READY

    def test_is_loaded(self, tmp_weights_dir: Path) -> None:
        """is_loaded() の初期状態は False."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        assert engine.is_loaded() is False

    @pytest.mark.asyncio
    async def test_load_unload(self, tmp_weights_dir: Path) -> None:
        """load/unload サイクル."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        assert not engine.is_loaded()
        await engine.load_model()
        assert engine.is_loaded()
        await engine.unload_model()
        assert not engine.is_loaded()

    @pytest.mark.asyncio
    async def test_reconstruct(self, tmp_weights_dir: Path, tmp_path: Path) -> None:
        """reconstruct() の基本動作."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        image = ProcessedImage(path=tmp_path / "test.jpg")
        params = ReconstructionParams(output_dir=tmp_path)

        progress_calls: list[tuple[float, str]] = []
        def cb(p: float, m: str) -> None:
            progress_calls.append((p, m))

        result = await engine.reconstruct([image], params, cb)

        assert result.success is True
        assert result.output_format == OutputFormat.OBJ
        assert len(progress_calls) == 2
        assert progress_calls[0] == (0.5, "halfway")
        assert progress_calls[1] == (1.0, "done")

    def test_output_formats(self, tmp_weights_dir: Path) -> None:
        """get_output_formats() が capabilities と一致."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        formats = engine.get_output_formats()
        assert OutputFormat.OBJ in formats
        assert OutputFormat.STL in formats

    def test_report_progress_none_callback(self, tmp_weights_dir: Path) -> None:
        """callback が None の場合エラーにならない."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        # 例外が発生しないことを確認
        engine._report_progress(None, 0.5, "test")

    def test_report_progress_error_callback(self, tmp_weights_dir: Path) -> None:
        """callback がエラーを起こしても例外にならない."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        def bad_cb(p: float, m: str) -> None:
            raise RuntimeError("callback error")
        # 例外が発生しないことを確認
        engine._report_progress(bad_cb, 0.5, "test")

    def test_report_progress_clamps(self, tmp_weights_dir: Path) -> None:
        """progress が 0.0〜1.0 にクランプされる."""
        engine = DummyEngine(weights_dir=tmp_weights_dir)
        received: list[float] = []
        def cb(p: float, _: str) -> None:
            received.append(p)
        engine._report_progress(cb, -0.5, "under")
        engine._report_progress(cb, 1.5, "over")
        assert received[0] == 0.0
        assert received[1] == 1.0
