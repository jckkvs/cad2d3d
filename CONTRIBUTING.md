# 開発に参加する

CAD3D Generator への貢献をお待ちしています！

## 開発環境のセットアップ

```bash
git clone https://github.com/jckkvs/cad2d3d.git
cd cad2d3d/backend
pip install -e ".[test]"

# テスト実行
python -m pytest tests/ -q
# → 248 passed

# カバレッジ付きテスト
python -m pytest tests/ --cov=app --cov-branch
```

## 開発のルール

### コーディングスタイル
- **ruff** で lint + format を統一
- **型注釈** を全ての公開関数に付ける
- **docstring** は日本語でOK、英語でもOK

```bash
# lint チェック
ruff check app/ tests/

# 自動フォーマット
ruff format app/ tests/
```

### テスト
- 新しい機能には必ずテストを追加
- テストファイル名: `tests/test_*.py`
- カバレッジ低下を避ける
- テストは `python -m pytest tests/ -q` で全合格すること

### コミットメッセージ
[Conventional Commits](https://www.conventionalcommits.org/) を使用:

```
feat: 新機能の追加
fix: バグ修正
docs: ドキュメントのみの変更
test: テストの追加・修正
refactor: リファクタリング
ci: CI設定の変更
```

## 新しいエンジンの追加方法

### 1. アダプタファイルを作成

`app/engines/<name>/adapter.py` を作成し、`ReconstructionEngine` を継承:

```python
from app.engines.base import ReconstructionEngine, ProcessedImage, ReconstructionParams, ReconstructionResult
from app.engines.registry import EngineRegistry
from app.models.schemas import EngineCapabilities, EngineStatus, OutputFormat, WeightFileInfo

@EngineRegistry.register
class MyEngine(ReconstructionEngine):
    def get_name(self) -> str:
        return "my_engine"

    def get_display_name(self) -> str:
        return "My Engine"

    def get_description(self) -> str:
        return "エンジンの説明"

    def get_version(self) -> str:
        return "1.0.0"

    def get_capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_single_image=True,
            outputs_mesh=True,
            supported_output_formats=[OutputFormat.GLB, OutputFormat.OBJ],
            requires_gpu=True,
            estimated_vram_gb=8.0,
        )

    def get_required_weights(self) -> list[WeightFileInfo]:
        return [
            WeightFileInfo(
                name="model.pth",
                url="https://huggingface.co/...",
                relative_path="my_engine/model.pth",
            )
        ]

    async def load_model(self) -> None:
        # 重みをロード
        self._model = ...

    async def unload_model(self) -> None:
        self._model = None

    async def reconstruct(self, images, params, progress_callback=None) -> ReconstructionResult:
        # 3D再構築ロジック
        ...
```

### 2. `__init__.py` を作成

```python
# app/engines/<name>/__init__.py
```

空ファイルでOK。`EngineRegistry.discover_engines()` が自動的にモジュールを発見します。

### 3. README_MODEL.md を作成（推奨）

```markdown
# My Engine

## 論文
- "Paper Title" (Author et al., 2024)

## パイプライン
1. 入力画像の前処理
2. モデル推論
3. メッシュ生成

## 重みファイル
- `model.pth`: 本体 (XXX MB)
```

### 4. テストを追加

`tests/test_engines_new.py` に以下を追加:

```python
def test_my_engine_info(self):
    info = self.get_engine("my_engine")
    assert info.name == "my_engine"
    assert info.capabilities.supports_single_image is True
```

### 5. プラグイン自動認識

上記が完了すれば、`EngineRegistry.discover_engines()` が自動検出し、
UIとAPIに自動的に表示されます。コード変更は不要です。

## プロジェクト構成の概要

```
backend/app/
├── engines/        ← エンジンアダプタ（プラグイン）
├── api/routes/     ← REST API
├── mold/           ← 金型設計計算
├── preprocessing/  ← 画像前処理
├── postprocessing/ ← メッシュ修復
├── similarity/     ← 類似度計算
├── history/        ← 履歴管理
├── models/         ← Pydantic スキーマ
└── core/           ← 設定・例外
```

## 質問・バグ報告

[GitHub Issues](https://github.com/jckkvs/cad2d3d/issues) でお気軽にどうぞ！
