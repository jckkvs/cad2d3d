"""
マルチビュー検出・分割モジュール.

1ファイル内に複数の視点画像（正面図・上面図・側面図）が含まれる場合の
自動検出・分割処理.

Implements: F-041 (マルチビュー分割)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


@dataclass
class DetectedView:
    """検出された視点領域."""
    image: Image.Image
    x: int
    y: int
    width: int
    height: int
    label: str = "unknown"  # front, top, side, isometric 等
    confidence: float = 0.0


def detect_and_split_views(
    image_path: Path,
    min_region_ratio: float = 0.05,
    gap_threshold: int = 30,
) -> list[DetectedView]:
    """
    画像内の複数ビューを検出して分割.

    空白領域（余白やガタースペース）を検出し、
    独立した図面ビューとして分割する.

    Args:
        image_path: 入力画像パス.
        min_region_ratio: 最小領域比率 (全体面積に対する比).
        gap_threshold: 空白ギャップ判定のピクセル閾値.

    Returns:
        検出されたビュー領域のリスト.
    """
    try:
        img = Image.open(image_path).convert("L")
    except Exception as e:
        raise PreprocessingError(f"画像の読み込みに失敗: {e}") from e

    arr = np.array(img)
    h, w = arr.shape
    total_area = h * w
    min_area = total_area * min_region_ratio

    # 二値化 (Otsu相当の簡易版)
    threshold = _otsu_threshold(arr)
    binary = (arr < threshold).astype(np.uint8)

    # 水平・垂直プロジェクションで空白帯を検出
    h_proj = binary.sum(axis=1)  # 各行のインク量
    v_proj = binary.sum(axis=0)  # 各列のインク量

    # 空白行/列の検出
    h_gaps = _find_gaps(h_proj, gap_threshold)
    v_gaps = _find_gaps(v_proj, gap_threshold)

    # 分割行と分割列からグリッドで領域を抽出
    row_ranges = _gaps_to_ranges(h_gaps, h)
    col_ranges = _gaps_to_ranges(v_gaps, w)

    views: list[DetectedView] = []
    original = Image.open(image_path).convert("RGB")

    for r_start, r_end in row_ranges:
        for c_start, c_end in col_ranges:
            region_area = (r_end - r_start) * (c_end - c_start)
            if region_area < min_area:
                continue

            # 領域内の実際のコンテンツを確認
            region_binary = binary[r_start:r_end, c_start:c_end]
            ink_ratio = region_binary.sum() / region_area
            if ink_ratio < 0.005:  # ほぼ空白の場合スキップ
                continue

            # 切り出し
            cropped = original.crop((c_start, r_start, c_end, r_end))
            views.append(DetectedView(
                image=cropped,
                x=c_start,
                y=r_start,
                width=c_end - c_start,
                height=r_end - r_start,
                confidence=ink_ratio,
            ))

    # ビューが検出されなかった場合は元画像をそのまま返す
    if not views:
        views.append(DetectedView(
            image=original,
            x=0, y=0, width=w, height=h,
            label="full",
            confidence=1.0,
        ))

    # ビューの位置からラベルを推測
    _assign_view_labels(views, w, h)

    logger.info("Detected %d views in %s", len(views), image_path.name)
    return views


def save_views(views: list[DetectedView], output_dir: Path, base_name: str) -> list[Path]:
    """検出されたビューを個別の画像ファイルに保存."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, view in enumerate(views):
        fname = f"{base_name}_view{i+1:02d}_{view.label}.png"
        out_path = output_dir / fname
        view.image.save(out_path, "PNG")
        paths.append(out_path)
        logger.info("Saved view: %s (%dx%d, label=%s)", out_path.name, view.width, view.height, view.label)
    return paths


def _otsu_threshold(arr: np.ndarray) -> int:
    """Otsu法による二値化閾値の計算."""
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = 0

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return best_threshold


def _find_gaps(projection: np.ndarray, threshold: int) -> list[tuple[int, int]]:
    """投影ヒストグラムから連続する空白ギャップを検出."""
    gaps: list[tuple[int, int]] = []
    in_gap = False
    gap_start = 0

    for i, val in enumerate(projection):
        if val <= threshold:
            if not in_gap:
                gap_start = i
                in_gap = True
        else:
            if in_gap:
                gaps.append((gap_start, i))
                in_gap = False

    if in_gap:
        gaps.append((gap_start, len(projection)))

    return gaps


def _gaps_to_ranges(gaps: list[tuple[int, int]], total_size: int) -> list[tuple[int, int]]:
    """ギャップリストから有効な範囲リストに変換."""
    if not gaps:
        return [(0, total_size)]

    ranges: list[tuple[int, int]] = []
    prev_end = 0

    for gap_start, gap_end in gaps:
        if gap_start > prev_end:
            ranges.append((prev_end, gap_start))
        prev_end = gap_end

    if prev_end < total_size:
        ranges.append((prev_end, total_size))

    return ranges


def _assign_view_labels(views: list[DetectedView], img_w: int, img_h: int) -> None:
    """
    ビューの位置関係からラベルを推測.

    一般的なエンジニアリング図面のレイアウト:
    - 左上: 正面図 (front)
    - 右上: 側面図 (right/side)
    - 左下: 上面図 (top) ※第三角法
    - 右下: 等角図 (isometric) [あれば]
    """
    if len(views) == 1:
        views[0].label = "unknown"
        return

    # 中心座標で分類
    cx_mid = img_w / 2
    cy_mid = img_h / 2

    for view in views:
        center_x = view.x + view.width / 2
        center_y = view.y + view.height / 2

        if center_x < cx_mid and center_y < cy_mid:
            view.label = "front"
        elif center_x >= cx_mid and center_y < cy_mid:
            view.label = "side"
        elif center_x < cx_mid and center_y >= cy_mid:
            view.label = "top"
        elif center_x >= cx_mid and center_y >= cy_mid:
            view.label = "isometric"
