# CAD3D Generator

[![Tests](https://img.shields.io/badge/tests-155%20passed-brightgreen)](https://github.com/jckkvs/cad2d3d)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> 2D図面・画像から3Dモデルを自動生成する統合プラットフォーム

## ✨ 特徴

- **9種類の変換エンジン** — TripoSR / Trellis / Hunyuan3D 2.0 / フォトグラメトリ / InstantMesh / CRM / Zero123++ / Wonder3D / SECAD-Net
- **エンジン比較機能** — 複数エンジンを並列実行し結果を比較
- **金型設計推定** — アンダーカット検出、パーティングライン最適化、型締力計算
- **NiceGUI + FastAPI** — モダンなWebインターフェース + REST API
- **3Dプレビュー** — ブラウザ内Three.jsビューア

## 🚀 クイックスタート

```bash
git clone https://github.com/jckkvs/cad2d3d.git
cd cad2d3d/backend
pip install -e ".[test]"

# テスト実行
python -m pytest tests/ -v

# NiceGUI起動
python nicegui_app.py
# → http://127.0.0.1:8080
```

## 🏗️ アーキテクチャ

```
backend/
├── app/
│   ├── engines/          # 9種の2D→3D変換エンジン
│   │   ├── base.py       # ReconstructionEngine ABC
│   │   ├── registry.py   # プラグイン自動検出
│   │   ├── triposr/      # TripoSR
│   │   ├── instantmesh/  # InstantMesh (10秒メッシュ生成)
│   │   ├── crm/          # CRM (6秒テクスチャ付き)
│   │   ├── secadnet/     # SECAD-Net (CADネイティブ出力)
│   │   └── ...
│   ├── api/routes/       # FastAPI エンドポイント
│   ├── mold/             # 金型設計 (undercut/PL/sizing)
│   ├── preprocessing/    # 画像前処理パイプライン
│   └── postprocessing/   # メッシュ修復
├── nicegui_app.py        # NiceGUI フロントエンド
└── tests/                # 155+ テスト
```

## 🔧 エンジン一覧

| エンジン | 入力 | 出力 | VRAM | 特徴 |
|---------|------|------|------|------|
| TripoSR | 画像 | メッシュ | 6GB | 高速単一画像 |
| Trellis | 画像 | メッシュ | 12GB | Microsoft SLAT |
| Hunyuan3D 2.0 | 画像 | メッシュ | 16GB | 高解像度 |
| フォトグラメトリ | 複数画像 | メッシュ/点群 | CPU | openMVG+openMVS |
| **InstantMesh** | 画像 | メッシュ | 10GB | 10秒生成 |
| **CRM** | 画像 | メッシュ | 6GB | 6秒テクスチャ付き |
| **Zero123++** | 画像 | メッシュ/点群 | 8GB | 6ビュー一貫生成 |
| **Wonder3D** | 画像 | メッシュ | 10GB | 法線活用高精度 |
| **SECAD-Net** | 3Dメッシュ | CAD (STEP) | 6GB | スケッチ-押出操作 |

## 🔬 金型設計機能

- アンダーカット検出 (レイキャスト+クラスタリング)
- パーティングライン最適化
- ドラフト角解析
- 型締力/サイクルタイム計算
- 標準部品データベース

## 📡 API

```
GET  /api/generate/engines           # エンジン一覧
POST /api/generate/run               # 3D生成
POST /api/generate/compare           # エンジン比較
GET  /api/mold/parts                 # 金型部品
POST /api/mold/sizing/clamp-force    # 型締力計算
```

## 🧪 テスト

```bash
# 全テスト
python -m pytest tests/ -v

# カバレッジ付き
python -m pytest tests/ --cov=app --cov-branch
```

## 📄 ドキュメント

- [REPRODUCEPROMPT.MD](REPRODUCEPROMPT.MD) — AI再現用の完全仕様書
- 各エンジンの `README_MODEL.md` — 論文引用+パイプライン説明
