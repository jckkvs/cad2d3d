# InstantMesh エンジン

## 概要
[InstantMesh](https://github.com/TencentARC/InstantMesh) はTencent ARCによる高速3Dメッシュ生成モデル。
単一画像から約10秒でテクスチャ付きメッシュを生成。

## パイプライン
```
入力画像 ──→ Zero123++ (マルチビュー生成)
           │ 6ビュー × 320x320
           ↓
     FlexiCubes (メッシュ抽出)
           │ 可微分等値面
           ↓
     テクスチャ付き3Dメッシュ (GLB/OBJ/STL)
```

## 論文
**"InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models"**
(Xu et al., 2024)

> "InstantMesh combines multi-view diffusion and sparse-view reconstruction to generate high-quality 3D meshes in ~10 seconds."

訳: InstantMeshはマルチビュー拡散とスパースビュー再構成を組み合わせ、約10秒で高品質3Dメッシュを生成。

## 要件
| 項目 | 値 |
|------|-----|
| GPU | CUDA 対応GPU必須 |
| VRAM | ≥ 10 GB |
| 入力 | 単一画像 (RGB) |
| 出力 | GLB / OBJ / STL / PLY |

## 使用方法
1. エンジンタブから "InstantMesh" を選択
2. 画像をアップロード
3. 「3Dモデルを生成」をクリック

## ライセンス
Apache-2.0
