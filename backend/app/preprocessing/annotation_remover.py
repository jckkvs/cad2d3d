"""
注釈・寸法線除去モジュール.

エンジニアリング図面から注釈テキスト、寸法線（矢印＋数値）、
補助線、ハッチングを検出・除去し、純粋な形状線のみを残す.

Implements: F-042 (注釈除去)

手法:
1. OpenCV: Hough 変換による直線検出 + 矢印パターン検出
2. OCR (EasyOCR/Tesseract): テキスト領域の検出とマスキング
3. Inpainting: 除去領域の修復
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.core.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


@dataclass
class AnnotationRegion:
    """検出された注釈領域."""
    x: int
    y: int
    width: int
    height: int
    annotation_type: str  # "text", "dimension_line", "arrow", "hatching", "auxiliary"
    text_content: str | None = None
    confidence: float = 0.0


@dataclass
class ScaleInfo:
    """検出されたスケール情報."""
    value: float | None = None
    unit: str = "mm"
    source_text: str = ""
    confidence: float = 0.0


@dataclass
class CleanupResult:
    """クリーンアップ処理の結果."""
    cleaned_image: Image.Image
    removed_annotations: list[AnnotationRegion] = field(default_factory=list)
    extracted_scale: ScaleInfo | None = None
    original_size: tuple[int, int] = (0, 0)


def remove_annotations(
    image_path: Path,
    remove_text: bool = True,
    remove_dimensions: bool = True,
    remove_hatching: bool = True,
    remove_auxiliary: bool = True,
    inpaint: bool = True,
) -> CleanupResult:
    """
    画像から注釈・寸法線・補助線を除去.

    Args:
        image_path: 入力画像パス.
        remove_text: テキスト注釈を除去するか.
        remove_dimensions: 寸法線(矢印+数値)を除去するか.
        remove_hatching: ハッチング(断面模様)を除去するか.
        remove_auxiliary: 補助線(一点鎖線等)を除去するか.
        inpaint: 除去跡をインペインティングで修復するか.

    Returns:
        クリーンアップ結果.
    """
    try:
        original = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise PreprocessingError(f"画像の読み込みに失敗: {e}") from e

    arr = np.array(original)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    annotations: list[AnnotationRegion] = []
    scale_info: ScaleInfo | None = None

    # 1. テキスト検出・除去
    if remove_text:
        text_regions, scale = _detect_text_regions(arr)
        for region in text_regions:
            annotations.append(region)
            _apply_mask(mask, region)
        if scale:
            scale_info = scale

    # 2. 寸法線検出・除去
    if remove_dimensions:
        dim_regions = _detect_dimension_lines(arr)
        for region in dim_regions:
            annotations.append(region)
            _apply_mask(mask, region)

    # 3. ハッチング検出・除去
    if remove_hatching:
        hatch_regions = _detect_hatching(arr)
        for region in hatch_regions:
            annotations.append(region)
            _apply_mask(mask, region)

    # 4. 補助線検出・除去
    if remove_auxiliary:
        aux_regions = _detect_auxiliary_lines(arr)
        for region in aux_regions:
            annotations.append(region)
            _apply_mask(mask, region)

    # 5. マスク領域のインペインティング
    if inpaint and mask.any():
        cleaned_arr = _inpaint_masked(arr, mask)
    else:
        cleaned_arr = arr.copy()
        cleaned_arr[mask > 0] = 255  # 白で塗りつぶし

    cleaned = Image.fromarray(cleaned_arr)

    logger.info(
        "Removed %d annotations from %s (text=%d, dim=%d, hatch=%d, aux=%d)",
        len(annotations), image_path.name,
        sum(1 for a in annotations if a.annotation_type == "text"),
        sum(1 for a in annotations if a.annotation_type in ("dimension_line", "arrow")),
        sum(1 for a in annotations if a.annotation_type == "hatching"),
        sum(1 for a in annotations if a.annotation_type == "auxiliary"),
    )

    return CleanupResult(
        cleaned_image=cleaned,
        removed_annotations=annotations,
        extracted_scale=scale_info,
        original_size=original.size,
    )


def _detect_text_regions(arr: np.ndarray) -> tuple[list[AnnotationRegion], ScaleInfo | None]:
    """
    OCRでテキスト領域を検出.

    EasyOCR → Tesseract の順にフォールバック.
    検出したテキストからスケール情報も抽出.
    """
    regions: list[AnnotationRegion] = []
    scale_info: ScaleInfo | None = None

    try:
        import easyocr
        reader = easyocr.Reader(["ja", "en"], gpu=False, verbose=False)
        results = reader.readtext(arr)

        for bbox, text, conf in results:
            if conf < 0.3:
                continue

            pts = np.array(bbox, dtype=np.int32)
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            margin = 5

            region = AnnotationRegion(
                x=max(0, x_min - margin),
                y=max(0, y_min - margin),
                width=min(arr.shape[1], x_max + margin) - max(0, x_min - margin),
                height=min(arr.shape[0], y_max + margin) - max(0, y_min - margin),
                annotation_type="text",
                text_content=text,
                confidence=conf,
            )
            regions.append(region)

            # スケール情報の抽出
            scale = _extract_scale_from_text(text, conf)
            if scale and (not scale_info or scale.confidence > scale_info.confidence):
                scale_info = scale

        return regions, scale_info

    except ImportError:
        pass

    try:
        import pytesseract
        gray = np.mean(arr, axis=2).astype(np.uint8) if arr.ndim == 3 else arr
        img_pil = Image.fromarray(gray)
        data = pytesseract.image_to_data(img_pil, lang="jpn+eng", output_type=pytesseract.Output.DICT)

        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            conf = float(data["conf"][i])
            if conf < 30:
                continue

            region = AnnotationRegion(
                x=data["left"][i],
                y=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
                annotation_type="text",
                text_content=text,
                confidence=conf / 100.0,
            )
            regions.append(region)

            scale = _extract_scale_from_text(text, conf / 100.0)
            if scale and (not scale_info or scale.confidence > scale_info.confidence):
                scale_info = scale

        return regions, scale_info

    except ImportError:
        logger.warning("OCRライブラリ(easyocr/pytesseract)がインストールされていません。テキスト検出をスキップします。")
        return regions, scale_info


def _extract_scale_from_text(text: str, confidence: float) -> ScaleInfo | None:
    """テキストからスケール情報を抽出."""
    import re

    # パターン: "100mm", "50 mm", "1:100", "Scale 1/50" 等
    patterns = [
        (r"(\d+(?:\.\d+)?)\s*(?:mm|ミリ)", "mm"),
        (r"(\d+(?:\.\d+)?)\s*(?:cm|センチ)", "cm"),
        (r"(\d+(?:\.\d+)?)\s*(?:m|メートル)", "m"),
        (r"(\d+(?:\.\d+)?)\s*(?:inch|in|インチ)", "inch"),
        (r"1\s*[:/]\s*(\d+)", "scale"),
        (r"R\s*(\d+(?:\.\d+)?)", "radius_mm"),
    ]

    for pattern, unit in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return ScaleInfo(
                value=float(match.group(1)),
                unit=unit,
                source_text=text,
                confidence=confidence,
            )

    return None


def _detect_dimension_lines(arr: np.ndarray) -> list[AnnotationRegion]:
    """
    OpenCVのHough変換で寸法線（矢印 + 数値）を検出.

    特徴: 形状線より細い / 矢印端点がある / 近くにテキストがある
    """
    regions: list[AnnotationRegion] = []

    try:
        import cv2

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr.copy()
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough変換で直線検出
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
        if lines is None:
            return regions

        # 水平・垂直に近い線をフィルタ (寸法線は typically 水平/垂直)
        # ただし対角寸法もあるので緩めの判定
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 20:
                continue

            # 線の角度
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            is_horizontal = angle < 10 or angle > 170
            is_vertical = 80 < angle < 100

            if not (is_horizontal or is_vertical):
                continue

            # 矢印端点の検出 (簡易: 線端付近のエッジ密度)
            endpoint_density = _check_arrow_endpoint(edges, x1, y1, x2, y2)
            if endpoint_density < 0.3:
                continue

            margin = 15
            x_min = max(0, min(x1, x2) - margin)
            y_min = max(0, min(y1, y2) - margin)
            x_max = min(arr.shape[1], max(x1, x2) + margin)
            y_max = min(arr.shape[0], max(y1, y2) + margin)

            regions.append(AnnotationRegion(
                x=x_min, y=y_min,
                width=x_max - x_min, height=y_max - y_min,
                annotation_type="dimension_line",
                confidence=endpoint_density,
            ))

    except ImportError:
        logger.warning("OpenCVがインストールされていません。寸法線検出をスキップします。")

    return regions


def _check_arrow_endpoint(
    edges: np.ndarray, x1: int, y1: int, x2: int, y2: int, radius: int = 10,
) -> float:
    """線の端点付近の矢印パターンを簡易チェック."""
    h, w = edges.shape

    def endpoint_density(cx: int, cy: int) -> float:
        y_start = max(0, cy - radius)
        y_end = min(h, cy + radius)
        x_start = max(0, cx - radius)
        x_end = min(w, cx + radius)
        region = edges[y_start:y_end, x_start:x_end]
        if region.size == 0:
            return 0.0
        return float(region.sum()) / (region.size * 255.0)

    d1 = endpoint_density(x1, y1)
    d2 = endpoint_density(x2, y2)
    return max(d1, d2)


def _detect_hatching(arr: np.ndarray) -> list[AnnotationRegion]:
    """
    ハッチング（断面パターン）の検出.

    特徴: 等間隔の平行線、通常は斜め45度.
    """
    regions: list[AnnotationRegion] = []

    try:
        import cv2

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr.copy()
        edges = cv2.Canny(gray, 30, 100)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=15, maxLineGap=5)
        if lines is None:
            return regions

        # 斜め45度 ± 15度の短い平行線を集約
        diagonal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if 30 < abs(angle) < 60 or 120 < abs(angle) < 150:
                diagonal_lines.append(line[0])

        if len(diagonal_lines) < 5:
            return regions

        # クラスタリング的に近接する斜線をグループ化
        diag_arr = np.array(diagonal_lines)
        centers = diag_arr.reshape(-1, 2, 2).mean(axis=1)  # 各線の中心点

        # 簡易クラスタ (グリッド分割)
        grid_size = 100
        clusters: dict[tuple[int, int], list] = {}
        for i, (cx, cy) in enumerate(centers):
            key = (int(cx) // grid_size, int(cy) // grid_size)
            clusters.setdefault(key, []).append(diagonal_lines[i])

        for key, lines_group in clusters.items():
            if len(lines_group) < 3:
                continue
            lines_arr = np.array(lines_group)
            x_min = max(0, lines_arr[:, [0, 2]].min() - 5)
            y_min = max(0, lines_arr[:, [1, 3]].min() - 5)
            x_max = min(arr.shape[1], lines_arr[:, [0, 2]].max() + 5)
            y_max = min(arr.shape[0], lines_arr[:, [1, 3]].max() + 5)

            regions.append(AnnotationRegion(
                x=int(x_min), y=int(y_min),
                width=int(x_max - x_min), height=int(y_max - y_min),
                annotation_type="hatching",
                confidence=min(1.0, len(lines_group) / 10.0),
            ))

    except ImportError:
        logger.warning("OpenCVが必要です。ハッチング検出をスキップします。")

    return regions


def _detect_auxiliary_lines(arr: np.ndarray) -> list[AnnotationRegion]:
    """
    補助線（一点鎖線・破線）の検出.

    特徴: 中心線は一点鎖線 - ・ - ・ -
    """
    # Phase 4で高精度版を実装予定
    # 現時点では空リストを返す（形状線を誤検出するリスクが高いため）
    return []


def _apply_mask(mask: np.ndarray, region: AnnotationRegion) -> None:
    """マスクに注釈領域を追加."""
    y_end = min(mask.shape[0], region.y + region.height)
    x_end = min(mask.shape[1], region.x + region.width)
    mask[region.y:y_end, region.x:x_end] = 255


def _inpaint_masked(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    マスク領域をインペインティングで修復.

    OpenCV の inpaint (Navier-Stokes / Telea)を使用.
    """
    try:
        import cv2
        result = cv2.inpaint(arr, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return result
    except ImportError:
        # OpenCV がない場合は白で塗りつぶし
        result = arr.copy()
        result[mask > 0] = 255
        return result
