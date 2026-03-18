# REPRODUCEPROMPT.MD — CAD3D Generator 完全再現プロンプト

> **目的**: 他の生成AIがこのプログラムを完全に再現できるよう、設計・実装・テストの全仕様を記述する。

---

## 1. プロジェクト概要

**CAD3D Generator** は、2D画像/CADファイルから3Dモデルを生成するWebアプリケーション。  
9種類の2D→3D変換エンジンを搭載し、ユーザーは複数エンジンの結果を比較できる。  
金型設計推定機能も統合。

### 技術スタック
- **バックエンド**: Python 3.13+, FastAPI, NiceGUI
- **3Dシーン**: NiceGUI `ui.scene` (Three.js)
- **MLフレームワーク**: PyTorch, trimesh, numpy
- **テスト**: pytest, pytest-cov, pytest-asyncio
- **パッケージ管理**: pip (pyproject.toml)

---

## 2. ディレクトリ構成

```
cad2d23d/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── generate.py          # 3D生成API (run/compare/engines/jobs)
│   │   │   ├── mold.py              # 金型設計API
│   │   │   ├── preprocess.py        # 前処理API
│   │   │   ├── settings.py          # 設定API
│   │   │   └── upload.py            # アップロードAPI
│   │   ├── core/
│   │   │   ├── config.py            # AppSettings (upload_dir, temp_dir等)
│   │   │   └── exceptions.py        # EngineNotFoundError等
│   │   ├── engines/
│   │   │   ├── base.py              # ReconstructionEngine ABC
│   │   │   ├── registry.py          # EngineRegistry (自動発見+登録)
│   │   │   ├── triposr/adapter.py   # TripoSR (既存)
│   │   │   ├── trellis/adapter.py   # Trellis (既存)
│   │   │   ├── hunyuan3d2/adapter.py # Hunyuan3D 2.0 (既存)
│   │   │   ├── photogrammetry/adapter.py # フォトグラメトリ (既存)
│   │   │   ├── instantmesh/adapter.py # InstantMesh (新規)
│   │   │   ├── crm/adapter.py       # CRM (新規)
│   │   │   ├── zero123pp/adapter.py # Zero123++ (新規)
│   │   │   ├── wonder3d/adapter.py  # Wonder3D (新規)
│   │   │   └── secadnet/adapter.py  # SECAD-Net (新規)
│   │   ├── mold/
│   │   │   ├── core.py              # MoldDesigner
│   │   │   ├── undercut.py          # アンダーカット検出
│   │   │   ├── parting_line.py      # パーティングライン最適化
│   │   │   ├── draft_analysis.py    # ドラフト角解析
│   │   │   ├── sizing.py            # 型締力/ランナー/冷却/鋼材選定
│   │   │   └── parts_db.py          # 標準部品DB
│   │   ├── models/schemas.py        # Pydanticスキーマ
│   │   ├── preprocessing/           # 画像前処理パイプライン
│   │   ├── postprocessing/          # メッシュ修復
│   │   ├── similarity/              # メッシュ/画像類似度
│   │   ├── weights/manager.py       # 重みダウンロード管理
│   │   ├── history/manager.py       # 生成履歴管理
│   │   └── main.py                  # FastAPI app
│   ├── nicegui_app.py               # NiceGUI フロントエンド
│   ├── tests/
│   │   ├── test_engines.py          # エンジン基盤テスト (24件)
│   │   ├── test_engines_new.py      # 新規エンジンテスト (47件)
│   │   ├── test_mold.py             # 金型テスト (65件)
│   │   └── test_api.py              # APIテスト
│   ├── pyproject.toml
│   └── static/viewer3d.html
└── frontend/                        # React (Vite) フロントエンド
```

---

## 3. エンジンアーキテクチャ

### 3.1 ReconstructionEngine ABC (`app/engines/base.py`)

```python
class ReconstructionEngine(ABC):
    @abstractmethod
    def get_name(self) -> str: ...
    @abstractmethod
    def get_display_name(self) -> str: ...
    @abstractmethod
    def get_description(self) -> str: ...
    @abstractmethod
    def get_version(self) -> str: ...
    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities: ...
    @abstractmethod
    def get_required_weights(self) -> list[WeightFile]: ...
    @abstractmethod
    async def load_model(self) -> None: ...
    @abstractmethod
    async def unload_model(self) -> None: ...
    @abstractmethod
    async def reconstruct(
        self, images: list[ProcessedImage],
        params: ReconstructionParams,
        progress_callback: Callable | None = None,
    ) -> ReconstructionResult: ...
    def is_loaded(self) -> bool: ...
    def check_ready(self) -> EngineStatus: ...
    def get_info(self) -> EngineInfo: ...
```

### 3.2 EngineRegistry (`app/engines/registry.py`)

```python
class EngineRegistry:
    @classmethod
    def register(cls, engine_class): ...     # デコレータ
    @classmethod
    def discover_engines(cls): ...           # pkgutil.walk_packages で自動発見
    @classmethod
    def get(cls, name: str) -> ReconstructionEngine: ...
    @classmethod
    def list_available(cls) -> list[EngineInfo]: ...
    @classmethod
    def reset(cls): ...                      # テスト用リセット
```

### 3.3 EngineCapabilities スキーマ

```python
@dataclass
class EngineCapabilities:
    supports_single_image: bool
    supports_multi_image: bool
    outputs_mesh: bool
    outputs_point_cloud: bool
    outputs_cad: bool
    supports_cad_input: bool
    requires_gpu: bool
    estimated_vram_gb: float
    supported_output_formats: list[OutputFormat]
```

---

## 4. 9エンジン仕様

### 4.1 TripoSR (既存)
- **入力**: 単一画像 → **出力**: メッシュ (GLB/OBJ/STL)
- **VRAM**: 6GB, **手法**: TripoSR transformer

### 4.2 Trellis (既存)
- **入力**: 単一画像 → **出力**: メッシュ (GLB/OBJ/STL/PLY)
- **VRAM**: 12GB, **手法**: Microsoft SLAT

### 4.3 Hunyuan3D 2.0 (既存)
- **入力**: 単一画像 → **出力**: メッシュ (GLB/OBJ/STL)
- **VRAM**: 16GB, **手法**: Tencent 2段階パイプライン

### 4.4 Photogrammetry (既存)
- **入力**: 複数画像 → **出力**: メッシュ/点群
- **GPU不要**, **手法**: openMVG + openMVS

### 4.5 InstantMesh (新規)
- **入力**: 単一画像 → **出力**: メッシュ (GLB/OBJ/STL/PLY)
- **VRAM**: 10GB (旧12GB → 修正)
- **手法**: Zero123++ マルチビュー → FlexiCubes メッシュ抽出
- **論文**: "InstantMesh: Efficient 3D Mesh Generation" (Xu et al., 2024)
- **重み**: `data/weights/instantmesh/instant_mesh_large.ckpt`

### 4.6 CRM (新規)
- **入力**: 単一画像 → **出力**: メッシュ (OBJ/GLB/STL)
- **VRAM**: 6GB
- **手法**: CCMs (Canonical Coordinate Maps) → FlexiCubes
- **論文**: "CRM: Single Image to 3D Textured Mesh" (Wang et al., 2024)
- **重み**: `data/weights/crm/CRM.pth`

### 4.7 Zero123++ (新規)
- **入力**: 単一画像 → **出力**: メッシュ/点群 (OBJ/STL/GLB/PLY)
- **VRAM**: 8GB
- **手法**: 1回のフォワードパスで6ビュー一貫生成 → SDF再構成
- **論文**: "Zero123++: Single Image to Consistent Multi-view Diffusion" (Shi et al., 2023)
- **重み**: `data/weights/zero123pp/unet/`, `data/weights/zero123pp/vae/`
- **バージョン**: "1.2.0"

### 4.8 Wonder3D (新規)
- **入力**: 単一画像 → **出力**: メッシュ (OBJ/STL/GLB/PLY)
- **VRAM**: 10GB
- **手法**: クロスドメイン拡散 → RGB+法線同時生成 → NeuS再構成
- **論文**: "Wonder3D: Single Image to 3D using Cross-Domain Diffusion" (Long et al., 2023)
- **重み**: `data/weights/wonder3d/unet/diffusion_pytorch_model.safetensors`

### 4.9 SECAD-Net (新規)
- **入力**: 3Dメッシュ → **出力**: CADネイティブ (STEP/OBJ/STL)
- **VRAM**: 6GB
- **手法**: PointNet++エンコーダ → SEデコーダ → スケッチ-押出操作列
- **論文**: "SECAD-Net: Self-Supervised CAD Reconstruction" (Li et al., 2023)
- **重み**: `data/weights/secadnet/encoder.pth`, `decoder.pth`
- **特殊データ構造**:
  - `SketchProfile`: 2D制御点(16点) + スケッチ平面(origin/normal)
  - `ExtrudeOperation`: タイプ(new_body/cut/join) + 深さ + 対称フラグ
  - `SketchExtrudeStep`: sketch + extrude + step_index
  - `SECADResult`: steps[] + num_operations + reconstruction_iou
- **ネットワーク**:
  - エンコーダ: Conv1D(3→64→128→256→512) + MaxPool → 512次元
  - デコーダ: FC(512→512→512→328) + FC(512→8) → 8操作 × 41パラメータ
  - IoU計算: ボクセル化(32³) → 占有率比較
- **ユーティリティ関数**:
  - `_build_pointnet_encoder()`: PointNet++構築
  - `_build_se_decoder()`: SEデコーダ構築
  - `_parse_se_operations(params, scale, center)`: パラメータ → SECADResult
  - `_se_operations_to_mesh(steps, scale, center)`: 操作列 → trimeshメッシュ
  - `_compute_iou(mesh_a, mesh_b)`: IoU計算

---

## 5. 比較API

### 5.1 エンドポイント

```
POST /api/generate/compare
  Body: GenerateRequest (engine_params.compare_engines: list[str])
  Response: { success, comparison_id, engine_jobs: { engine: job_id } }

GET /api/generate/compare/{job_ids}
  job_ids: カンマ区切りジョブID
  Response: { success, all_complete, results: { job_id: { status, result } } }

GET /api/generate/engines
  Response: list[EngineInfo]

GET /api/generate/engines/{engine_name}
  Response: EngineInfo

POST /api/generate/run
  Response: { success, job_id }

GET /api/generate/jobs/{job_id}
  Response: { status, progress }

GET /api/generate/jobs/{job_id}/result
  Response: { success, result }
```

---

## 6. NiceGUI フロントエンド (`nicegui_app.py`)

### 6.1 レイアウト
```
┌──────────────────────────────────────────────────────┐
│ ◆ CAD3D Generator  v0.1.0        NiceGUI Edition  ⚙ │
├──────────┬───────────────────────────────────────────┤
│ タブパネル │        3D プレビュー (Three.js)            │
│          │                                           │
│ [入力]    │    ┌───────────────────────────┐          │
│ [エンジン] │    │   ui.scene (800x600)      │          │
│ [比較]    │    │   グリッド + モデル表示     │          │
│ [重み]    │    └───────────────────────────┘          │
│ [金型]    │    [💾DL] [🔗外部] [🔄リロード]             │
└──────────┴───────────────────────────────────────────┘
```

### 6.2 比較タブ
- エンジン一覧をチェックボックスで表示 (名前 + ステータス + VRAM)
- 「⚡比較生成を開始」ボタン → `run_comparison_generation()` 呼び出し
- 並列生成後、結果を生成時間順のカードグリッドで表示
- 各カードに: エンジン名 + 生成時間 + メタデータ + DLボタン

### 6.3 注意点 (Python 3.13)
- NiceGUI `@ui.page` 内のasync関数では、イベントハンドラを**lambda**でラップして参照
  - `on_click=lambda: download_result()` ✅ (遅延評価)
  - `on_click=download_result` ❌ (定義前参照でUnboundLocalError)
- `refresh_weights_ui` 等の遅延呼び出しは `ui.timer(0.1, ..., once=True)` を使用

---

## 7. 金型設計モジュール (`app/mold/`)

### 7.1 コンポーネント
| ファイル | 機能 |
|---------|------|
| `undercut.py` | アンダーカット検出 (レイキャスト+クラスタリング) |
| `parting_line.py` | パーティングライン最適化 (候補方向スコアリング) |
| `draft_analysis.py` | ドラフト角解析 (面法線→抜き勾配計算) |
| `sizing.py` | サイジング (型締力/ランナー/冷却/突出/鋼材/サイクルタイム) |
| `parts_db.py` | 標準部品DB (ガイドピン/突出ピン/ガススプリング等) |
| `core.py` | MoldDesigner (全機能統合) |

### 7.2 API (`app/api/routes/mold.py`)
```
POST /api/mold/design         # 全体設計
POST /api/mold/undercut       # アンダーカット解析
GET  /api/mold/parts          # 部品一覧
GET  /api/mold/sizing/clamp-force  # 型締力計算
```

---

## 8. テスト仕様

### 8.1 テスト構成 (248件合格, カバレッジ60%)
| ファイル | 件数 | 内容 |
|---------|------|------|
| `test_engines.py` | 24 | ReconstructionEngine ABC, EngineRegistry |
| `test_engines_new.py` | 47 | 新規5エンジン + SECAD-Netユーティリティ + 比較API + unload + 整合性 |
| `test_mold.py` | 65 | アンダーカット/PL/ドラフト/サイジング/部品DB/API統合 |
| `test_api_integration.py` | 19 | FastAPI全エンドポイント統合テスト |
| `test_modules.py` | 23 | 前処理/メッシュ修復/重み管理/類似度 |
| `test_history.py` | 15 | 履歴管理CRUD/永続化/破損回復 |
| `test_coverage_boost.py` | 35 | Config/例外/Export/Settings/スキーマ/エンジンエッジケース |
| `test_generate_preprocessing.py` | 20 | Generate API/Multiview分割/Otsu/ギャップ検出 |

### 8.2 テスト初期化の注意
```python
# test_engines_new.py のsession fixtureで全エンジンを強制登録
@pytest.fixture(autouse=True, scope="session")
def _setup_engine_registry():
    EngineRegistry.reset()
    for mod_name in engine_modules:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)  # デコレータ再発火に必須
```
- `EngineRegistry` の `_discovered` フラグが一度Trueになると再発見しない
- テスト間で順序依存を避けるため `reset()` + `reload()` が必要

### 8.3 カバレッジ (60%, 分岐含む)
| モジュール | カバレッジ |
|-----------|----------|
| `engines/base.py` | **100%** |
| `models/schemas.py` | **100%** |
| `core/config.py` | **100%** |
| `core/exceptions.py` | **100%** |
| `history/manager.py` | **99%** |
| `engines/registry.py` | **92%** |
| `mold/sizing.py` | **91%** |
| `multiview_splitter.py` | **88%** |
| `settings.py` | **93%** |
- `load_model` / `reconstruct` 内部はGPU+重み依存のため未テスト

---

## 9. 依存関係 (`pyproject.toml`)

```toml
[project]
name = "cad3d-generator"
dependencies = [
    "fastapi>=0.100",
    "uvicorn[standard]",
    "nicegui>=1.4",
    "torch>=2.0",
    "trimesh>=4.0",
    "numpy>=1.24",
    "Pillow>=10.0",
    "huggingface-hub",
    "httpx",
    "pydantic>=2.0",
]

[project.optional-dependencies]
test = ["pytest", "pytest-asyncio", "pytest-cov", "pytest-xdist"]
```

---

## 10. セットアップ・実行手順

```bash
# 1. クローン
git clone https://github.com/jckkvs/cad2d3d.git
cd cad2d3d/backend

# 2. 依存インストール
pip install -e ".[test]"

# 3. テスト実行 (248件)
python -m pytest tests/ -v

# カバレッジ付き
python -m pytest tests/ --cov=app --cov-branch

# 4. NiceGUI起動
python nicegui_app.py
# → http://127.0.0.1:8080

# 5. FastAPI単体起動
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 11. 再現時の重要注意

1. **エンジン登録デコレータ**: 各adapter.pyの末尾で `@EngineRegistry.register` を使用
2. **SECAD-Net パラメータ次元**: デコーダ出力 = 328 (8操作 × 41パラメータ/操作)
3. **IoU計算**: `trimesh.contains` は非水密メッシュで不正確 → アサーションは `0.0 <= iou <= 1.0`
4. **NiceGUI Python 3.13**: on_click は必ず lambda でラップ
5. **テスト順序非依存**: session fixture で `EngineRegistry.reset()` + `importlib.reload()`
6. **ポート競合**: NiceGUI は `show=True` でブラウザ自動起動 → テスト時は `show=False`
