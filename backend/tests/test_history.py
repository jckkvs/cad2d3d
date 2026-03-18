"""
生成履歴管理モジュールのテスト.

テスト対象:
- HistoryManager の CRUD 操作
- インデックスファイルの永続化
- 結果ファイルのコピー・削除
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# HistoryManager は settings.data_dir に依存するため、
# テスト用にdata_dirをパッチして使用する


class TestHistoryManager:
    """HistoryManager のテスト."""

    @pytest.fixture
    def history_mgr(self, tmp_path: Path):
        """テスト用HistoryManagerを作成."""
        with patch("app.history.manager.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            from app.history.manager import HistoryManager
            mgr = HistoryManager()
            yield mgr

    @pytest.fixture
    def sample_output(self, tmp_path: Path) -> Path:
        """テスト用出力ファイル."""
        out = tmp_path / "output.glb"
        out.write_bytes(b"FAKE_GLB_DATA_12345")
        return out

    def test_initial_state(self, history_mgr) -> None:
        """初期状態で履歴は空."""
        assert history_mgr.list_history() == []

    def test_save_result(self, history_mgr, sample_output: Path) -> None:
        """結果の保存."""
        entry = history_mgr.save_result(
            job_id="test-job-001",
            engine_name="triposr",
            input_files=["image.png"],
            output_path=sample_output,
            output_format="glb",
            elapsed_seconds=5.2,
            metadata={"resolution": 256},
        )
        assert entry["job_id"] == "test-job-001"
        assert entry["engine_name"] == "triposr"
        assert entry["elapsed_seconds"] == 5.2
        assert entry["metadata"]["resolution"] == 256
        assert "timestamp" in entry

    def test_list_history(self, history_mgr, sample_output: Path) -> None:
        """履歴一覧."""
        history_mgr.save_result("j1", "crm", [], sample_output, "obj", 1.0)
        history_mgr.save_result("j2", "trellis", [], sample_output, "glb", 2.0)
        history_mgr.save_result("j3", "triposr", [], sample_output, "stl", 3.0)

        items = history_mgr.list_history()
        assert len(items) == 3
        # 新しい順（j3が先頭）
        assert items[0]["job_id"] == "j3"
        assert items[2]["job_id"] == "j1"

    def test_list_history_limit(self, history_mgr, sample_output: Path) -> None:
        """履歴のlimit指定."""
        for i in range(10):
            history_mgr.save_result(f"j{i}", "crm", [], sample_output, "obj", 1.0)

        assert len(history_mgr.list_history(limit=5)) == 5
        assert len(history_mgr.list_history(limit=20)) == 10

    def test_get_entry(self, history_mgr, sample_output: Path) -> None:
        """ジョブIDで履歴取得."""
        history_mgr.save_result("my-job", "trellis", ["a.png"], sample_output, "glb", 4.0)
        entry = history_mgr.get_entry("my-job")
        assert entry is not None
        assert entry["engine_name"] == "trellis"

    def test_get_entry_not_found(self, history_mgr) -> None:
        """存在しないジョブID."""
        assert history_mgr.get_entry("nonexistent") is None

    def test_delete_entry(self, history_mgr, sample_output: Path) -> None:
        """履歴削除."""
        history_mgr.save_result("del-me", "crm", [], sample_output, "obj", 1.0)
        assert history_mgr.get_entry("del-me") is not None

        result = history_mgr.delete_entry("del-me")
        assert result is True
        assert history_mgr.get_entry("del-me") is None

    def test_delete_entry_not_found(self, history_mgr) -> None:
        """存在しない履歴の削除."""
        assert history_mgr.delete_entry("nonexistent") is False

    def test_get_result_path(self, history_mgr, sample_output: Path) -> None:
        """結果ファイルパスの取得."""
        history_mgr.save_result("path-test", "triposr", [], sample_output, "glb", 2.0)
        path = history_mgr.get_result_path("path-test")
        assert path is not None
        assert path.exists()
        assert path.name == "output.glb"

    def test_get_result_path_not_found(self, history_mgr) -> None:
        """存在しないジョブの結果パス."""
        assert history_mgr.get_result_path("nonexistent") is None

    def test_file_copy(self, history_mgr, sample_output: Path) -> None:
        """出力ファイルが履歴ディレクトリにコピーされる."""
        history_mgr.save_result("copy-test", "crm", [], sample_output, "glb", 1.0)
        copied = history_mgr._history_dir / "copy-test" / "output.glb"
        assert copied.exists()
        assert copied.read_bytes() == b"FAKE_GLB_DATA_12345"

    def test_persistence(self, tmp_path: Path) -> None:
        """インデックスの永続化."""
        with patch("app.history.manager.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            from app.history.manager import HistoryManager

            # 1回目: 保存
            mgr1 = HistoryManager()
            out = tmp_path / "test.obj"
            out.write_text("fake mesh")
            mgr1.save_result("persist-1", "crm", [], out, "obj", 1.0)
            mgr1.save_result("persist-2", "trellis", [], out, "obj", 2.0)

            # 2回目: 再読込
            mgr2 = HistoryManager()
            items = mgr2.list_history()
            assert len(items) == 2
            assert items[0]["job_id"] == "persist-2"

    def test_save_with_nonexistent_output(self, history_mgr, tmp_path: Path) -> None:
        """存在しない出力ファイルでもエントリは保存される."""
        fake_path = tmp_path / "nonexistent.stl"
        entry = history_mgr.save_result("no-file", "crm", [], fake_path, "stl", 0.5)
        assert entry["job_id"] == "no-file"
        assert history_mgr.get_entry("no-file") is not None

    def test_corrupted_index_recovery(self, tmp_path: Path) -> None:
        """破損したインデックスファイルからの回復."""
        with patch("app.history.manager.settings") as mock_settings:
            mock_settings.data_dir = tmp_path
            history_dir = tmp_path / "history"
            history_dir.mkdir()
            index_file = history_dir / "index.json"
            index_file.write_text("NOT VALID JSON {{{", encoding="utf-8")

            from app.history.manager import HistoryManager
            mgr = HistoryManager()
            # 破損インデックスを空リストとして処理
            assert mgr.list_history() == []

    def test_metadata_optional(self, history_mgr, sample_output: Path) -> None:
        """metadataなしで保存."""
        entry = history_mgr.save_result("no-meta", "crm", [], sample_output, "obj", 1.0)
        assert entry["metadata"] == {}
