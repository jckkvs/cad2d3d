# SECAD-Net エンジン

## 概要
SECAD-Net は自己教師ありで Sketch-Extrude 操作を学習し、
3DメッシュからCADネイティブな表現を再構成するモデル。
**金型設計との親和性が最も高い**エンジン。

## パイプライン
```
3Dメッシュ ──→ 点群サンプリング (10,000点)
             │
             ↓
       PointNet++ エンコーダ
             │ → 512次元潜在コード
             ↓
       SE デコーダ
             │ → K個の Sketch-Extrude 操作列
             ↓
       CADソリッド構築
             │ スケッチ+押出 → ブーリアン合成
             ↓
       CADネイティブ表現 (STEP/OBJ/STL)
```

## 論文
**"SECAD-Net: Self-Supervised CAD Reconstruction by Learning Sketch-Extrude Operations"**
(Li et al., 2023) — Computers & Graphics

> "SECAD-Net learns to reconstruct a CAD solid from a raw 3D shape by discovering sketch-extrude operations without supervision."

訳: SECAD-Netは教師なしでスケッチ-押出操作を発見し、生の3D形状からCADソリッドを再構成する。

> "Each sketch-extrude (SE) operation consists of a 2D closed sketch profile on the sketch plane and a 1D extrude specification including the extrusion type and the extrusion extent."

訳: 各SE操作は、スケッチ平面上の2D閉じたスケッチプロファイルと、押出タイプと深さを含む1D押出仕様からなる。

## ネットワーク構成

| コンポーネント | 構造 | 出力 |
|-------------|------|------|
| エンコーダ | PointNet++ (Conv1D×4 + MaxPool) | 512次元潜在コード |
| デコーダ | FC×3 + 操作数予測ヘッド | K×41パラメータ |
| 操作パラメータ | 16×2制御点 + 3原点 + 3法線 + 深さ + タイプ + 対称 | 41次元/操作 |

## 要件
| 項目 | 値 |
|------|-----|
| GPU | 必須 |
| VRAM | ≥ 6 GB |
| 入力 | 3Dメッシュ (STL/OBJ/GLB) |
| 出力 | STEP / OBJ / STL |

## 特徴
- **CADネイティブ**: 編集可能なパラメトリック表現を出力
- **金型親和**: スケッチ→押出→切削 = 金型設計の基本操作
- **自己教師**: 教師データ不要で操作を自動発見
- **IoU評価**: 入力形状との一致度を自動計算

## 金型設計との関連
SECAD-Netの出力はスケッチ+押出操作のシーケンスであり、
これは金型設計における以下のワークフローと直接対応:

1. **スケッチ** → 製品断面形状
2. **押出 (new_body)** → キャビティ形状
3. **押出 (cut)** → スライドコア/アンダーカット処理
4. **操作列** → 金型加工手順の自動生成

## 参考文献
- hal-04164264v1 / S0097849323000766
