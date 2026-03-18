# CAD3D Generator

**2D画像/CADから3D CADモデルを生成**するオフライン対応Webアプリケーション。

画像データは一切オンラインに送信されません。すべての処理はローカルで実行されます。

## 特徴

- 🖼️ **多形式入力**: JPG, PNG, PDF, DXF, SVG, TIFF, HEIC 等
- 🔌 **プラグインアーキテクチャ**: 生成エンジンを簡単に追加・差し替え
- 🚀 **最先端エンジン**: TripoSR, Trellis, Img2CAD 等に対応予定
- 🔒 **オフライン処理**: 機密画像は外部に送信しない
- 🔧 **CAD連携**: 外部CADソフトで開く / 再読み込み
- 📦 **重み管理**: ワンクリックDL, プロキシ, HuggingFace Token 対応

## セットアップ

### バックエンド (Python)

```bash
cd backend
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

### フロントエンド (Node.js)

```bash
cd frontend
npm install
npm run dev
```

ブラウザで http://localhost:5173 を開いてください。

## アーキテクチャ

```
backend/   → FastAPI (Python)  - REST API + WebSocket
frontend/  → Vite + React + TypeScript - Three.js 3Dビューア
```

### エンジンの追加方法

1. `backend/app/engines/` にフォルダを作成
2. `ReconstructionEngine` を継承したクラスを実装
3. `@EngineRegistry.register` デコレータを付与
4. `README_MODEL.md` を配置

自動的にUI・APIに反映されます。

## ライセンス

MIT
