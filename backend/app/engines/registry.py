"""
エンジンレジストリ.

Strategy + Registryパターンで、利用可能なエンジンの登録・発見・取得を管理する.
新しいエンジンは @EngineRegistry.register デコレータで自動登録される.

Implements: F-010 (エンジン選択) | 設計: §4.2 プラグインアーキテクチャ
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.exceptions import EngineNotFoundError
from app.models.schemas import EngineInfo

if TYPE_CHECKING:
    from app.engines.base import ReconstructionEngine

logger = logging.getLogger(__name__)


class EngineRegistry:
    """
    エンジンレジストリ (Singleton).

    使い方:
    ```python
    # エンジン登録 (デコレータ)
    @EngineRegistry.register
    class MyEngine(ReconstructionEngine):
        ...

    # エンジン一覧取得
    engines = EngineRegistry.list_available()

    # エンジン取得
    engine = EngineRegistry.get("my_engine")
    ```
    """

    _engine_classes: dict[str, type[ReconstructionEngine]] = {}
    _engine_instances: dict[str, ReconstructionEngine] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, engine_cls: type[ReconstructionEngine]) -> type[ReconstructionEngine]:
        """
        エンジンクラスを登録するデコレータ.

        Args:
            engine_cls: ReconstructionEngine のサブクラス.

        Returns:
            登録されたクラス (デコレータなのでそのまま返す).
        """
        # クラスを仮インスタンス化して名前を取得
        # (weights_dir はダミー; 名前取得のみ)
        temp_instance = engine_cls.__new__(engine_cls)
        # __init__ を呼ばずに get_name を得るために、クラス変数的に名前を取得
        # → get_name は abstractmethod なので、実際のインスタンスが必要
        # → 安全な方法: weights_dir にダミーを渡してインスタンス化
        try:
            temp = engine_cls(weights_dir=settings.weights_dir)
            name = temp.get_name()
        except Exception as e:
            # get_name() がファイルI/O等を含む場合のフォールバック
            name = engine_cls.__name__.lower().replace("engine", "")
            logger.warning(
                "Could not get engine name from %s, using fallback '%s': %s",
                engine_cls.__name__,
                name,
                e,
            )

        if name in cls._engine_classes:
            logger.warning(
                "Engine '%s' is already registered (class: %s). Overwriting.",
                name,
                cls._engine_classes[name].__name__,
            )

        cls._engine_classes[name] = engine_cls
        logger.info("Registered engine: %s (%s)", name, engine_cls.__name__)
        return engine_cls

    @classmethod
    def discover_engines(cls) -> None:
        """
        engines/ パッケージ以下のモジュールを自動インポートし、
        @register されたエンジンを発見する.
        """
        if cls._discovered:
            return

        engines_package = "app.engines"
        engines_dir = Path(__file__).parent

        logger.info("Discovering engines in %s ...", engines_dir)

        for _importer, modname, ispkg in pkgutil.walk_packages(
            path=[str(engines_dir)],
            prefix=engines_package + ".",
        ):
            if modname.endswith((".base", ".registry")):
                continue
            try:
                importlib.import_module(modname)
                logger.debug("Imported engine module: %s", modname)
            except Exception:
                logger.warning("Failed to import engine module: %s", modname, exc_info=True)

        cls._discovered = True
        logger.info("Engine discovery complete. Found %d engines.", len(cls._engine_classes))

    @classmethod
    def get(cls, name: str) -> ReconstructionEngine:
        """
        名前でエンジンインスタンスを取得.

        同一名前のエンジンは1つのインスタンスを再利用する.

        Args:
            name: エンジン識別子.

        Returns:
            ReconstructionEngine インスタンス.

        Raises:
            EngineNotFoundError: 指定名のエンジンが登録されていない場合.
        """
        cls.discover_engines()

        if name not in cls._engine_classes:
            available = list(cls._engine_classes.keys())
            raise EngineNotFoundError(
                f"Engine '{name}' not found. Available: {available}",
                detail=f"利用可能なエンジン: {', '.join(available) or '(なし)'}",
            )

        if name not in cls._engine_instances:
            engine_cls = cls._engine_classes[name]
            cls._engine_instances[name] = engine_cls(weights_dir=settings.weights_dir)
            logger.info("Created engine instance: %s", name)

        return cls._engine_instances[name]

    @classmethod
    def list_available(cls) -> list[EngineInfo]:
        """
        登録済みエンジンの情報一覧を返す.

        Returns:
            EngineInfo のリスト.
        """
        cls.discover_engines()

        infos: list[EngineInfo] = []
        for name in sorted(cls._engine_classes.keys()):
            try:
                engine = cls.get(name)
                infos.append(engine.get_info())
            except Exception:
                logger.warning("Failed to get info for engine '%s'", name, exc_info=True)

        return infos

    @classmethod
    def reset(cls) -> None:
        """
        レジストリをリセット (テスト用).
        """
        cls._engine_classes.clear()
        cls._engine_instances.clear()
        cls._discovered = False
