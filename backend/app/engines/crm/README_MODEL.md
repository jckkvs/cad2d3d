# CRM エンジン

## 概要
[CRM](https://github.com/thu-ml/CRM) は清華大学による超高速3Dメッシュ生成モデル。
**わずか6秒**でテクスチャ付きメッシュを生成。

## パイプライン
```
入力画像 ──→ CCM拡散 (Canonical Coordinate Maps)
           │ 6ビュー座標マップ
           ↓
     FlexiCubes 再構成
           │ 幾何+テクスチャ同時生成
           ↓
     テクスチャ付き3Dメッシュ (GLB/OBJ/STL)
```

## 論文
**"CRM: Single Image to 3D Textured Mesh with Convolutional Reconstruction Model"**
(Wang et al., 2024)

> "CRM first generates multi-view canonical coordinate maps and then employs a differentiable renderer with FlexiCubes to reconstruct textured 3D meshes."

訳: CRMはマルチビュー正準座標マップを生成し、FlexiCubesでテクスチャ付き3Dメッシュを再構成。

## 要件
| 項目 | 値 |
|------|-----|
| GPU | 必須 |
| VRAM | ≥ 6 GB |
| 入力 | 単一画像 (RGB) |
| 出力 | OBJ / GLB / STL |

## ライセンス
Apache-2.0
