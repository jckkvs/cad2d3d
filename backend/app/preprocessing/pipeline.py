"""
画像前処理パイプライン.

フォーマット変換 → マルチビュー分割 → 注釈除去 の統合パイプライン.

Implements: F-040〜F-043
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.core.exceptions import PreprocessingError
from app.preprocessing.format_converter import convert_to_images
from app.preprocessing.multiview_splitter import (
    DetectedView,
    detect_and_split_views,
    save_views,
)
from app.preprocessing.annotation_remover import (
    AnnotationRegion,
    CleanupResult,
    ScaleInfo,
    remove_annotations,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedImage:
    """前処理済み画像の情報."""
    path: Path
    original_path: Path
    view_label: str = "unknown"
    scale_info: ScaleInfo | None = None
    removed_annotations: list[AnnotationRegion] = field(default_factory=list)
    was_split: bool = False


@dataclass
class PreprocessingConfig:
    """前処理設定."""
    auto_split_views: bool = True
    remove_text: bool = True
    remove_dimensions: bool = True
    remove_hatching: bool = True
    remove_auxiliary: bool = True
    inpaint: bool = True


def preprocess_file(
    input_path: Path,
    output_dir: Path,
    config: PreprocessingConfig | None = None,
) -> list[PreprocessedImage]:
    """
    入力ファイルの完全な前処理パイプライン.

    1. フォーマット変換 (PDF/SVG/DXF→PNG)
    2. マルチビュー検出・分割
    3. 注釈・寸法線除去

    Args:
        input_path: 入力ファイルパス.
        output_dir: 出力ディレクトリ.
        config: 前処理設定.

    Returns:
        前処理済み画像のリスト.
    """
    if config is None:
        config = PreprocessingConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[PreprocessedImage] = []

    # Step 1: フォーマット変換
    logger.info("Step 1: Format conversion for %s", input_path.name)
    try:
        converted_paths = convert_to_images(input_path)
    except PreprocessingError:
        raise
    except Exception as e:
        raise PreprocessingError(f"フォーマット変換に失敗: {e}") from e

    for conv_path in converted_paths:
        page_results = _process_single_image(conv_path, input_path, output_dir, config)
        results.extend(page_results)

    logger.info("Preprocessing complete: %s -> %d images", input_path.name, len(results))
    return results


def _process_single_image(
    image_path: Path,
    original_path: Path,
    output_dir: Path,
    config: PreprocessingConfig,
) -> list[PreprocessedImage]:
    """単一画像の前処理."""
    results: list[PreprocessedImage] = []

    # Step 2: マルチビュー検出・分割
    views: list[DetectedView] = []
    was_split = False

    if config.auto_split_views:
        logger.info("Step 2: Multi-view detection for %s", image_path.name)
        views = detect_and_split_views(image_path)
        was_split = len(views) > 1
        if was_split:
            logger.info("Detected %d views in %s", len(views), image_path.name)
    
    if not views:
        img = Image.open(image_path).convert("RGB")
        views = [DetectedView(image=img, x=0, y=0, width=img.width, height=img.height)]

    # Step 3: 各ビューの注釈除去
    for i, view in enumerate(views):
        view_name = f"{image_path.stem}_v{i+1:02d}_{view.label}"

        # 一時ファイルに保存
        temp_view_path = output_dir / f"{view_name}_raw.png"
        view.image.save(temp_view_path, "PNG")

        # 注釈除去
        if config.remove_text or config.remove_dimensions or config.remove_hatching:
            logger.info("Step 3: Annotation removal for %s", view_name)
            try:
                cleanup = remove_annotations(
                    temp_view_path,
                    remove_text=config.remove_text,
                    remove_dimensions=config.remove_dimensions,
                    remove_hatching=config.remove_hatching,
                    remove_auxiliary=config.remove_auxiliary,
                    inpaint=config.inpaint,
                )
                # クリーン画像を保存
                clean_path = output_dir / f"{view_name}_clean.png"
                cleanup.cleaned_image.save(clean_path, "PNG")

                results.append(PreprocessedImage(
                    path=clean_path,
                    original_path=original_path,
                    view_label=view.label,
                    scale_info=cleanup.extracted_scale,
                    removed_annotations=cleanup.removed_annotations,
                    was_split=was_split,
                ))
            except PreprocessingError:
                # 注釈除去に失敗した場合は元画像を使用
                logger.warning("Annotation removal failed, using raw view", exc_info=True)
                results.append(PreprocessedImage(
                    path=temp_view_path,
                    original_path=original_path,
                    view_label=view.label,
                    was_split=was_split,
                ))
        else:
            results.append(PreprocessedImage(
                path=temp_view_path,
                original_path=original_path,
                view_label=view.label,
                was_split=was_split,
            ))

        # 一時ファイルのクリーンアップ
        if temp_view_path.exists() and (output_dir / f"{view_name}_clean.png").exists():
            temp_view_path.unlink(missing_ok=True)

    return results
