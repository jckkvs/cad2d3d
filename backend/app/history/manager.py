"""
生成履歴管理モジュール.

プロジェクト単位での生成結果の保存・バージョン管理.

Implements: F-062 (生成履歴)
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class HistoryManager:
    """生成履歴の管理."""

    def __init__(self) -> None:
        self._history_dir = settings.data_dir / "history"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._history_dir / "index.json"
        self._index: list[dict[str, Any]] = self._load_index()

    def _load_index(self) -> list[dict[str, Any]]:
        """インデックスファイルを読み込み."""
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _save_index(self) -> None:
        """インデックスファイルを保存."""
        self._index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def save_result(
        self,
        job_id: str,
        engine_name: str,
        input_files: list[str],
        output_path: Path,
        output_format: str,
        elapsed_seconds: float,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        生成結果を履歴に保存.

        Args:
            job_id: ジョブID.
            engine_name: 使用したエンジン名.
            input_files: 入力ファイル名リスト.
            output_path: 出力ファイルパス.
            output_format: 出力フォーマット.
            elapsed_seconds: 処理時間.
            metadata: 追加メタデータ.

        Returns:
            保存された履歴エントリ.
        """
        # 結果ファイルを履歴ディレクトリにコピー
        entry_dir = self._history_dir / job_id
        entry_dir.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            dst = entry_dir / output_path.name
            shutil.copy2(str(output_path), str(dst))

        entry = {
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "engine_name": engine_name,
            "input_files": input_files,
            "output_file": output_path.name,
            "output_format": output_format,
            "elapsed_seconds": elapsed_seconds,
            "metadata": metadata or {},
        }

        self._index.insert(0, entry)  # 新しい順
        self._save_index()
        logger.info("Saved history entry: %s", job_id)
        return entry

    def list_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """履歴一覧を返す."""
        return self._index[:limit]

    def get_entry(self, job_id: str) -> dict[str, Any] | None:
        """指定ジョブの履歴を取得."""
        for entry in self._index:
            if entry["job_id"] == job_id:
                return entry
        return None

    def delete_entry(self, job_id: str) -> bool:
        """指定ジョブの履歴を削除."""
        entry_dir = self._history_dir / job_id
        if entry_dir.exists():
            shutil.rmtree(entry_dir)

        original_len = len(self._index)
        self._index = [e for e in self._index if e["job_id"] != job_id]
        if len(self._index) < original_len:
            self._save_index()
            logger.info("Deleted history entry: %s", job_id)
            return True
        return False

    def get_result_path(self, job_id: str) -> Path | None:
        """履歴の結果ファイルパスを取得."""
        entry = self.get_entry(job_id)
        if not entry:
            return None
        path = self._history_dir / job_id / entry["output_file"]
        return path if path.exists() else None


# シングルトン
history_manager = HistoryManager()
