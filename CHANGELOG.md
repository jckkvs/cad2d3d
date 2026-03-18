# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-03-19

### Added
- **5 新エンジン**: InstantMesh, CRM, Zero123++, Wonder3D, SECAD-Net
- **エンジン比較機能**: 複数エンジンの並列実行と結果比較（API + NiceGUI UI）
- **248テスト** (8テストファイル): エンジン基盤/金型/API/前処理/履歴/スキーマ
- **60% コードカバレッジ** (分岐カバレッジ含む)
- **GitHub Actions CI**: Python 3.11/3.12/3.13 + ruff lint
- **REPRODUCEPROMPT.MD**: AI再現用の完全仕様書
- **CONTRIBUTING.md**: 開発者ガイド + エンジン追加手順
- **README_MODEL.md × 5**: 各新エンジンの論文引用 + パイプライン説明
- **CADソフト互換表**: SolidWorks/Fusion/Blender/FreeCAD/AutoCAD/CATIA/NX
- **金型設計詳細ドキュメント**: 9計算モジュール + 10樹脂 + 計算根拠(Kazmer/Rees)
- **フォトグラメトリ撮影ガイド**: 最低枚数/撮影角度/重なり/照明
- **メッシュ品質情報**: エンジン別の頂点数/テクスチャ/水密性
- **類似度機能ドキュメント**: D2 Shape Distribution + DNN埋め込み
- **プロキシ設定ガイド**: 企業内ネットワーク対応
- **Mermaid ワークフロー図**: 入力→エンジン→出力の全体フロー

### Fixed
- **NiceGUI 500エラー**: Python 3.13のクロージャスコープ問題
  (`download_result`/`open_external`/`reload_model`をUI構築前に定義)
- **テスト安定性**: conftest.py fixture scope を `session`→`module` に変更
- **メッシュ修復**: trimeshバージョン互換 (`remove_self_intersections=False`)
- **画像類似度**: 均一色画像でのedge_similarity=0対応
- **Pydanticスキーマ**: APIResponse.error属性テスト修正, EngineInfo.required_weights必須化

### Changed
- **README.md**: 開発者向け → 導入者/ユーザー/開発者 3層向けに大幅リライト
- **入出力形式**: 3形式記載 → 全7出力形式 + 全13入力形式を明記

## [0.1.0] - 2026-03-17

### Added
- 初期リリース
- 4エンジン: TripoSR, Trellis, Hunyuan3D 2.0, フォトグラメトリ
- 金型設計モジュール: アンダーカット検出, パーティングライン, サイジング
- NiceGUI フロントエンド + FastAPI バックエンド
- 3Dプレビュー (Three.js)
