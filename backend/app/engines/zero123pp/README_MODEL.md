# Zero123++ エンジン

## 概要
[Zero123++](https://github.com/SUDO-AI-3D/zero123plus) はStability AIによる一貫マルチビュー拡散モデル。
1回のフォワードパスで6つの一貫したビューを同時生成。

## パイプライン
```
入力画像 ──→ Zero123++ 拡散モデル
           │ 1回のフォワードパス
           ↓
     6ビュー一貫画像
           │ 正面/右/左/上/背面/下
           ↓
     マルチビューステレオ / SDF再構成
           ↓
     3Dメッシュ (OBJ/STL/GLB/PLY)
```

## 論文
**"Zero123++: a Single Image to Consistent Multi-view Diffusion Base Model"**
(Shi et al., 2023)

> "Zero123++ generates six consistent multi-view images from a single input image in a single forward pass."

訳: 1回のフォワードパスで6つの一貫したマルチビュー画像を生成。

## 要件
| 項目 | 値 |
|------|-----|
| GPU | 必須 |
| VRAM | ≥ 8 GB |
| 入力 | 単一画像 (RGB) |
| 出力 | OBJ / STL / GLB / PLY |

## 特徴
- **一貫性**: 6ビュー間の幾何整合性を保証
- **点群対応**: PLY出力で点群としても利用可能
- **下流タスク**: InstantMeshやNeuSのマルチビュー入力として利用

## ライセンス
Apache-2.0
