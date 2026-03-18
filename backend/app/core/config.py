"""
アプリケーション設定.
環境変数または .env ファイルから読み込む.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """アプリケーション全体の設定."""

    app_name: str = "CAD3D Generator"
    app_version: str = "0.1.0"
    debug: bool = False

    # ファイルパス
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = base_dir / "data"
    weights_dir: Path = data_dir / "weights"
    projects_dir: Path = data_dir / "projects"
    temp_dir: Path = data_dir / "temp"
    upload_dir: Path = data_dir / "uploads"

    # アップロード制限
    max_upload_size_mb: int = 200
    allowed_image_extensions: list[str] = [
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".webp", ".heic", ".heif", ".svg",
    ]
    allowed_cad_extensions: list[str] = [
        ".dxf", ".dwg", ".step", ".stp", ".iges", ".igs",
    ]
    allowed_document_extensions: list[str] = [".pdf"]

    # ネットワーク設定
    http_proxy: str | None = None
    https_proxy: str | None = None
    hf_token: str | None = None
    hf_cache_dir: Path | None = None

    # サーバー設定
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    model_config = {"env_prefix": "CAD3D_", "env_file": ".env"}

    def ensure_dirs(self) -> None:
        """必要なディレクトリを作成."""
        for d in [
            self.data_dir,
            self.weights_dir,
            self.projects_dir,
            self.temp_dir,
            self.upload_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
