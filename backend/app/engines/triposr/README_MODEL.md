# TripoSR — 単一画像→3Dメッシュ高速生成エンジン

## 概要

**TripoSR** は Stability AI と Tripo AI が共同開発した、単一画像から3Dメッシュを高速生成するオープンソースモデルです。

- **論文**: [TripoSR: Fast 3D Object Reconstruction from a Single Image](https://arxiv.org/abs/2402.02459) (2024)
- **GitHub**: https://github.com/VAST-AI-Research/TripoSR
- **ライセンス**: MIT

## 動作要件

| 項目 | 最小要件 | 推奨 |
|---|---|---|
| GPU | NVIDIA (VRAM 4GB+) | NVIDIA (VRAM 8GB+) |
| CUDA | 11.7+ | 12.1+ |
| Python | 3.10+ | 3.11 |
| OS | Windows / Linux | - |

## インストール手順

### 1. PyTorch のインストール

```bash
# CUDA 12.1 の場合
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8 の場合
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. TripoSR のインストール

```bash
pip install triposr
```

## モデル重みファイル

### 自動ダウンロード

アプリの「重み管理」画面から「ダウンロード」ボタンを押すと自動でダウンロードされます。

### 手動配置

ダウンロードできない場合、以下の手順で手動配置してください。

1. HuggingFace からモデルをダウンロード:
   - URL: https://huggingface.co/stabilityai/TripoSR
   - ファイル: `model.ckpt`, `config.yaml`

2. 以下のディレクトリに配置:
   ```
   data/weights/triposr/
   ├── model.ckpt
   └── config.yaml
   ```

### プロキシ環境での設定

プロキシ環境の場合、アプリの「設定」→「高度設定」からプロキシを設定できます。

```
HTTP プロキシ:  http://proxy.example.com:8080
HTTPS プロキシ: http://proxy.example.com:8080
```

### HuggingFace トークン

一部のモデルは認証が必要です。HuggingFace のアクセストークンを設定してください:
1. https://huggingface.co/settings/tokens でトークンを作成
2. アプリの「設定」→「高度設定」→「HuggingFace Token」にペースト

## 入出力仕様

### 入力

- **画像**: 1枚 (JPG, PNG 等)
- **推奨**: 背景が単色の物体写真

### 出力フォーマット

| 形式 | 拡張子 | 説明 |
|---|---|---|
| OBJ | .obj | メッシュ + テクスチャ座標 |
| STL | .stl | 3Dプリント用 |
| GLB | .glb | Web3D用 (バイナリ) |
| glTF | .gltf | Web3D用 (JSON) |
| PLY | .ply | 点群 / メッシュ |

## トラブルシューティング

### CUDA out of memory

解像度を下げるか、`resolution` パラメータを 128 に設定してください。

### モデルが見つからない

`data/weights/triposr/` に `model.ckpt` が配置されていることを確認してください。
