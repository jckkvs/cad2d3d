"""
カスタム例外定義.
"""


class CAD3DError(Exception):
    """アプリケーション基底例外."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(self.message)


class EngineNotFoundError(CAD3DError):
    """指定されたエンジンが見つからない."""


class EngineNotReadyError(CAD3DError):
    """エンジンの準備が完了していない (重みが未DL等)."""


class WeightDownloadError(CAD3DError):
    """モデル重みのダウンロード失敗."""


class FileFormatError(CAD3DError):
    """サポートされていないファイル形式."""


class ProcessingError(CAD3DError):
    """画像前処理または3D生成中のエラー."""


class PreprocessingError(CAD3DError):
    """前処理パイプラインのエラー."""


class ProjectNotFoundError(CAD3DError):
    """指定されたプロジェクトが見つからない."""

