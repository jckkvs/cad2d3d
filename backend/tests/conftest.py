"""
共有テストフィクスチャ.

全テストファイルで使用可能なフィクスチャを定義。
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient


# ── エンジンレジストリ初期化 ─────────────────────

ENGINE_MODULES = [
    "app.engines.triposr.adapter",
    "app.engines.trellis.adapter",
    "app.engines.hunyuan3d2.adapter",
    "app.engines.photogrammetry.adapter",
    "app.engines.instantmesh.adapter",
    "app.engines.crm.adapter",
    "app.engines.zero123pp.adapter",
    "app.engines.wonder3d.adapter",
    "app.engines.secadnet.adapter",
]


def _ensure_engines_registered():
    """レジストリをリセットして全エンジンを強制登録."""
    from app.engines.registry import EngineRegistry
    EngineRegistry.reset()
    for mod_name in ENGINE_MODULES:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)


@pytest.fixture(scope="module")
def ensure_engines():
    """エンジンレジストリを初期化するフィクスチャ (module scope).

    test_engines.py は独自の reset_registry fixture を持つため、
    このフィクスチャは test_engines_new.py 等で明示的に使用。
    """
    _ensure_engines_registered()
    yield


# ── API テストクライアント ─────────────────────

@pytest.fixture(scope="session")
def api_client() -> TestClient:
    """FastAPI TestClient を返す."""
    from app.main import app
    return TestClient(app)


# ── テスト用メッシュ ──────────────────────────

@pytest.fixture
def simple_box() -> trimesh.Trimesh:
    """10x10x10 ボックスメッシュ."""
    return trimesh.creation.box(extents=(10, 10, 10))


@pytest.fixture
def small_box() -> trimesh.Trimesh:
    """5x5x5 ボックスメッシュ."""
    return trimesh.creation.box(extents=(5, 5, 5))


@pytest.fixture
def cylinder_mesh() -> trimesh.Trimesh:
    """円柱メッシュ (半径5, 高さ20)."""
    return trimesh.creation.cylinder(radius=5, height=20)


@pytest.fixture
def l_shape_mesh() -> trimesh.Trimesh:
    """L字型メッシュ (アンダーカットテスト用)."""
    box_a = trimesh.creation.box(extents=(20, 10, 10))
    box_b = trimesh.creation.box(extents=(10, 10, 20))
    box_b.apply_translation([5, 0, 15])
    return trimesh.boolean.union([box_a, box_b], engine="scad")


@pytest.fixture
def tmp_mesh_path(simple_box, tmp_path) -> Path:
    """一時STLファイルパス."""
    path = tmp_path / "test_box.stl"
    simple_box.export(str(path))
    return path


@pytest.fixture
def tmp_image_path(tmp_path) -> Path:
    """一時テスト画像パス (100x100 白画像)."""
    from PIL import Image
    path = tmp_path / "test_image.png"
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    img.save(str(path))
    return path


# ── エンジン名一覧 ───────────────────────────

ALL_ENGINE_NAMES = [
    "triposr", "trellis", "hunyuan3d2", "photogrammetry",
    "instantmesh", "crm", "zero123pp", "wonder3d", "secadnet",
]

NEW_ENGINE_NAMES = [
    "instantmesh", "crm", "zero123pp", "wonder3d", "secadnet",
]


@pytest.fixture(params=ALL_ENGINE_NAMES)
def engine_name(request) -> str:
    """パラメタライズされたエンジン名."""
    return request.param


@pytest.fixture(params=NEW_ENGINE_NAMES)
def new_engine_name(request) -> str:
    """パラメタライズされた新エンジン名."""
    return request.param
