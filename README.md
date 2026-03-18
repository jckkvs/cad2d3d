# 🏭 CAD3D Generator

[![Tests](https://img.shields.io/badge/tests-248%20passed-brightgreen)](https://github.com/jckkvs/cad2d3d)
[![Coverage](https://img.shields.io/badge/coverage-60%25-yellowgreen)](https://github.com/jckkvs/cad2d3d)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13-blue)](https://python.org)

> **2D図面や写真をアップロードするだけで、3Dモデルを自動生成する**統合プラットフォーム

---

## 🎯 こんな方に

| ユーザー | できること |
|---------|----------|
| **設計者** | 手描きスケッチや2D CAD図面から3Dモデルを素早く作成 |
| **金型エンジニア** | 3Dモデルからアンダーカット検出・パーティングライン最適化・型締力計算を自動推定 |
| **開発者** | REST API経由で3D生成機能を自社システムに組み込み |
| **研究者** | 9種類のAIエンジンの出力品質を一画面で比較 |

---

## 📸 画面イメージ

起動すると、ブラウザで以下のようなインターフェースが表示されます：

```
┌──────────────────────────────────────────────────────────┐
│  ◆ CAD3D Generator  v0.1.0         NiceGUI Edition  ⚙   │
├──────────┬───────────────────────────────────────────────┤
│ [入力]    │           3D プレビュー (Three.js)             │
│ [エンジン] │     ┌─────────────────────────────┐           │
│ [比較]    │     │                             │           │
│ [重み]    │     │    回転・ズーム対応の         │           │
│ [金型]    │     │    リアルタイム3Dビューア     │           │
│          │     │                             │           │
│          │     └─────────────────────────────┘           │
│          │     [💾 ダウンロード] [🔗 外部CADで開く]         │
└──────────┴───────────────────────────────────────────────┘
```

**5つのタブ**で全機能にアクセスできます：

| タブ | 機能 |
|-----|------|
| **入力** | 画像/CADファイルのアップロード、プレビュー |
| **エンジン** | 9種類の変換エンジンから選択、ワンクリック変換 |
| **比較** | 複数エンジンを並列実行し、生成時間・品質を比較 |
| **重み** | AIモデルの重みダウンロード状況の確認・管理 |
| **金型** | アンダーカット検出・型締力計算・部品選定 |

---

## 🚀 導入手順

### 前提条件

| 項目 | 最低要件 | 推奨 |
|------|---------|------|
| **OS** | Windows 10 / macOS / Linux | Windows 11 |
| **Python** | 3.11 以上 | 3.13 |
| **メモリ** | 8GB | 16GB以上 |
| **GPU** | なしでも動作（フォトグラメトリ） | NVIDIA GPU (VRAM 6GB+) |
| **ディスク** | 2GB (コードのみ) | 20GB (重みファイル含む) |

> **💡 GPU がなくても使えます！**  
> フォトグラメトリエンジンはCPUのみで動作します。  
> GPU搭載PCでは、TripoSR/CRM等のAIエンジンも利用可能になります。

### インストール

```bash
# 1. リポジトリをクローン
git clone https://github.com/jckkvs/cad2d3d.git
cd cad2d3d/backend

# 2. パッケージをインストール（テストツール含む）
pip install -e ".[test]"

# 3. 動作確認（248件のテストが全て合格すればOK）
python -m pytest tests/ -q
# → 248 passed ✅
```

### 起動

```bash
# NiceGUI（Web UI）を起動
python nicegui_app.py
```

ブラウザで **http://127.0.0.1:8080** が自動的に開きます。

### API だけ使いたい場合

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/docs でSwagger UIが開きます
```

---

## 🔧 使い方

### 基本の流れ

```
1. 「入力」タブで画像をアップロード
2. 「エンジン」タブでお好みのエンジンを選択
3. 「生成」ボタンをクリック
4. 3Dプレビューで結果を確認
5. GLB/OBJ/STL形式でダウンロード
```

### エンジン比較

```
1. 「比較」タブで複数のエンジンにチェック
2. 「⚡比較生成を開始」をクリック
3. 全エンジンが並列実行され、結果が一覧表示
4. 生成時間・品質を見比べて最適なエンジンを選定
```

### 金型設計

```
1. 3Dモデルを生成（またはアップロード）
2. 「金型」タブでアンダーカット解析を実行
3. パーティングラインの最適化を確認
4. 型締力・サイクルタイム・標準部品を自動推定
```

---

## 🔧 エンジン一覧

| エンジン | 特徴 | GPU | 速度 | おすすめ用途 |
|---------|------|-----|------|------------|
| **TripoSR** | 写真1枚から生成 | 6GB | ⚡高速 | まず試したい方 |
| **CRM** | テクスチャ付き | 6GB | ⚡⚡最速 | テクスチャ重視 |
| **InstantMesh** | 高品質メッシュ | 10GB | ⚡高速 | 品質重視 |
| **Zero123++** | 6面同時生成 | 8GB | 中 | 全方位の整合性 |
| **Wonder3D** | 法線活用で高精度 | 10GB | 中 | 細部の精度 |
| **Trellis** | Microsoft技術 | 12GB | 中 | 大規模モデル |
| **Hunyuan3D** | 高解像度 | 16GB | 遅 | 最高品質 |
| **フォトグラメトリ** | 複数写真から | CPUのみ✅ | 遅 | GPU不要 |
| **SECAD-Net** | メッシュ→CAD変換 | 6GB | 中 | CADデータ化 |

> **💡 初めての方へ**: まず **CRM** か **TripoSR** をお試しください（軽量・高速）

---

## 📡 REST API リファレンス

起動後に `http://localhost:8000/docs` でインタラクティブなAPIドキュメントを確認できます。

### 主要エンドポイント

| メソッド | パス | 説明 |
|---------|------|------|
| `GET` | `/api/generate/engines` | エンジン一覧 |
| `GET` | `/api/generate/engines/{name}` | エンジン詳細 |
| `POST` | `/api/generate/run` | 3D生成ジョブの開始 |
| `POST` | `/api/generate/compare` | 複数エンジンを並列比較 |
| `GET` | `/api/generate/jobs/{id}` | ジョブの進捗確認 |
| `GET` | `/api/export/download/{id}` | 結果ファイルのダウンロード |
| `POST` | `/api/mold/sizing/clamp-force` | 型締力計算 |
| `GET` | `/api/mold/parts` | 金型標準部品一覧 |

---

## 🏗️ プロジェクト構成

```
cad2d3d/
├── backend/
│   ├── app/
│   │   ├── engines/          # 9種の2D→3D変換エンジン
│   │   ├── api/routes/       # FastAPI エンドポイント
│   │   ├── mold/             # 金型設計モジュール
│   │   ├── preprocessing/    # 画像前処理(ビュー分割/注釈除去)
│   │   ├── postprocessing/   # メッシュ修復
│   │   ├── similarity/       # 3D/2D類似度比較
│   │   └── history/          # 生成履歴管理
│   ├── nicegui_app.py        # NiceGUI フロントエンド
│   └── tests/                # 248テスト
├── .github/workflows/ci.yml  # GitHub Actions CI
├── REPRODUCEPROMPT.MD         # AI再現用 完全仕様書
└── README.md                  # ← このファイル
```

---

## ❓ トラブルシューティング

### アプリが起動しない

```bash
# ポート8080が使用中の場合
python nicegui_app.py --port 8081
```

### エンジンが「WEIGHTS_MISSING」と表示される

AIエンジンは初回利用時に重みファイルのダウンロードが必要です。  
「重み」タブからダウンロードするか、以下のディレクトリに手動配置してください：

```
backend/data/weights/
├── triposr/
├── instantmesh/
├── crm/
└── ...
```

### GPU が認識されない

```bash
# PyTorchのGPUサポートを確認
python -c "import torch; print(torch.cuda.is_available())"
# → True なら利用可能

# False の場合、CUDA対応版PyTorchを再インストール
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### テストが失敗する

```bash
# 詳細ログ付きで実行
python -m pytest tests/ -v --tb=long

# カバレッジ付き
python -m pytest tests/ --cov=app --cov-branch
```

---

## 🧪 開発者向け

### テスト実行

```bash
python -m pytest tests/ -v          # 全248テスト
python -m pytest tests/ -q          # サマリだけ
python -m pytest tests/ --cov=app   # カバレッジ付き
```

### CI

GitHub Actions で Python 3.11 / 3.12 / 3.13 に対してテスト+lintが自動実行されます。

### 新しいエンジンの追加

1. `app/engines/<name>/adapter.py` を作成
2. `ReconstructionEngine` を継承して必須メソッドを実装
3. ファイル末尾で `@EngineRegistry.register` デコレータを適用
4. テストを `tests/test_engines_new.py` に追加

→ 自動的にUIとAPIに表示されます（プラグインアーキテクチャ）

---

## 📄 関連ドキュメント

- [REPRODUCEPROMPT.MD](REPRODUCEPROMPT.MD) — AI再現用の完全仕様書（他のAIがこのプロジェクトを再構築するための詳細仕様）
- 各エンジンの `README_MODEL.md` — 論文引用+パイプライン説明
- `/docs` — Swagger UI（起動後にアクセス可能）

---

## 📜 ライセンス

MIT License
