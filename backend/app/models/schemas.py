"""
Pydanticスキーマ / データモデル定義.
APIリクエスト・レスポンスの型安全性を保証する.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


# ── 列挙型 ──────────────────────────────────────────────


class ViewAngle(str, Enum):
    """カメラアングル指定."""
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    ISOMETRIC = "isometric"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class OutputFormat(str, Enum):
    """3D出力フォーマット."""
    STL = "stl"
    OBJ = "obj"
    PLY = "ply"
    GLTF = "gltf"
    GLB = "glb"
    STEP = "step"
    IGES = "iges"


class EngineStatus(str, Enum):
    """エンジン準備状態."""
    READY = "ready"
    WEIGHTS_MISSING = "weights_missing"
    DEPENDENCY_MISSING = "dependency_missing"
    ERROR = "error"


class JobStatus(str, Enum):
    """ジョブ状態."""
    QUEUED = "queued"
    PREPROCESSING = "preprocessing"
    GENERATING = "generating"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── エンジン関連 ─────────────────────────────────────────


class WeightFileInfo(BaseModel):
    """モデル重みファイル情報."""
    name: str
    url: str
    size_bytes: int | None = None
    sha256: str | None = None
    relative_path: str  # weights_dir からの相対パス
    description: str = ""
    requires_auth: bool = False  # HuggingFace gated model 等


class EngineCapabilities(BaseModel):
    """エンジンの能力情報."""
    supports_single_image: bool = False
    supports_multi_image: bool = False
    supports_cad_input: bool = False
    outputs_mesh: bool = False
    outputs_cad: bool = False
    outputs_point_cloud: bool = False
    supported_output_formats: list[OutputFormat] = []
    requires_gpu: bool = False
    estimated_vram_gb: float | None = None


class EngineInfo(BaseModel):
    """エンジン概要情報 (UI表示用)."""
    name: str
    display_name: str
    description: str
    version: str
    capabilities: EngineCapabilities
    status: EngineStatus
    required_weights: list[WeightFileInfo]
    readme_path: str | None = None


# ── ファイルアップロード ──────────────────────────────────


class UploadedFileInfo(BaseModel):
    """アップロード済みファイル情報."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_name: str
    stored_path: str
    file_size: int
    mime_type: str | None = None
    uploaded_at: datetime = Field(default_factory=datetime.now)


class ImageWithAngle(BaseModel):
    """アングル情報付き画像."""
    file_id: str
    view_angle: ViewAngle = ViewAngle.UNKNOWN
    custom_azimuth: float | None = None  # degree
    custom_elevation: float | None = None  # degree


# ── 3D生成ジョブ ─────────────────────────────────────────


class GenerationRequest(BaseModel):
    """3D生成リクエスト."""
    engine_name: str
    images: list[ImageWithAngle]
    output_format: OutputFormat = OutputFormat.GLB
    engine_params: dict = Field(default_factory=dict)


class GenerationProgress(BaseModel):
    """3D生成進捗 (WebSocket送信用)."""
    job_id: str
    status: JobStatus
    progress: float = 0.0  # 0.0 ~ 1.0
    message: str = ""
    error: str | None = None


class GenerationResult(BaseModel):
    """3D生成結果."""
    job_id: str
    status: JobStatus
    output_file: str | None = None
    output_format: OutputFormat | None = None
    preview_url: str | None = None
    elapsed_seconds: float = 0.0
    engine_name: str = ""
    metadata: dict = Field(default_factory=dict)


# ── 設定関連 ─────────────────────────────────────────────


class ProxySettings(BaseModel):
    """プロキシ設定."""
    http_proxy: str | None = None
    https_proxy: str | None = None
    no_proxy: str | None = None


class HuggingFaceSettings(BaseModel):
    """HuggingFace設定."""
    token: str | None = None
    cache_dir: str | None = None


class AppSettings(BaseModel):
    """アプリ設定 (UI表示/更新用)."""
    proxy: ProxySettings = ProxySettings()
    huggingface: HuggingFaceSettings = HuggingFaceSettings()


# ── 汎用レスポンス ───────────────────────────────────────


class APIResponse(BaseModel):
    """汎用APIレスポンス."""
    success: bool
    message: str = ""
    data: dict | list | None = None
