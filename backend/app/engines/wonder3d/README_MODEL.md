# Wonder3D エンジン

## 概要
[Wonder3D](https://github.com/xxlong0/Wonder3D) はクロスドメイン拡散で
RGB画像と法線マップを同時に生成し、NeuSベースの高品質表面再構成を実現。

## パイプライン
```
入力画像 ──→ クロスドメイン拡散
           │ Cross-Domain Attention
           ├── 6ビュー RGB 画像
           └── 6ビュー 法線マップ
                ↓
          NeuS 表面再構成
           │ RGB + Normal → SDF → メッシュ
           ↓
     高精度3Dメッシュ (OBJ/STL/GLB/PLY)
```

## 論文
**"Wonder3D: Single Image to 3D using Cross-Domain Diffusion"**
(Long et al., 2023)

> "Wonder3D produces consistent multi-view normal maps and the corresponding color images by leveraging cross-domain attention."

訳: クロスドメインアテンションにより一貫したマルチビュー法線マップとカラー画像を同時生成。

## 要件
| 項目 | 値 |
|------|-----|
| GPU | 必須 |
| VRAM | ≥ 10 GB |
| 入力 | 単一画像 (RGB) |
| 出力 | OBJ / STL / GLB / PLY |

## 特徴
- **法線活用**: 法線マップで微細な幾何形状を再現
- **高精度**: NeuSベースのSDF再構成で滑らかな表面
- **CAD向き**: 金型設計等で必要な高精度ジオメトリ

## ライセンス
研究目的
