"""
深度マップ推定モジュール.

単一画像から深度マップを推定し、3D再構築の精度向上に活用.

対応モデル:
- Depth Anything V2 (Apple/HuggingFace)
- MiDaS (Intel ISL)

Implements: F-060 (深度推定)
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


async def estimate_depth(
    image_path: Path,
    output_dir: Path,
    model_name: str = "depth_anything_v2",
) -> Path:
    """
    単一画像から深度マップを推定.

    Args:
        image_path: 入力画像パス.
        output_dir: 出力ディレクトリ.
        model_name: 使用するモデル ("depth_anything_v2" or "midas").

    Returns:
        深度マップ画像のパス.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_depth.png"

    if model_name == "depth_anything_v2":
        return await _depth_anything_v2(image_path, output_path)
    elif model_name == "midas":
        return await _midas(image_path, output_path)
    else:
        raise PreprocessingError(f"未対応の深度推定モデル: {model_name}")


async def _depth_anything_v2(image_path: Path, output_path: Path) -> Path:
    """Depth Anything V2 による深度推定."""
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )

        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # 正規化して画像に変換
        depth = predicted_depth.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth_img = Image.fromarray(depth.astype(np.uint8))
        depth_img = depth_img.resize(img.size, Image.BILINEAR)
        depth_img.save(output_path)

        logger.info("Depth map estimated: %s", output_path)
        return output_path

    except ImportError:
        raise PreprocessingError(
            "Depth Anything V2 には transformers + torch が必要です",
            detail="pip install transformers torch",
        )


async def _midas(image_path: Path, output_path: Path) -> Path:
    """MiDaS (Intel ISL) による深度推定."""
    try:
        import torch

        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = transforms.small_transform

        img = np.array(Image.open(image_path).convert("RGB"))
        input_batch = transform(img)

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        depth_img = Image.fromarray(depth.astype(np.uint8))
        depth_img.save(output_path)

        return output_path

    except ImportError:
        raise PreprocessingError(
            "MiDaS には torch が必要です",
            detail="pip install torch",
        )


def depth_to_point_cloud(
    depth_path: Path,
    image_path: Path,
    output_path: Path,
    focal_length: float = 500.0,
) -> Path:
    """
    深度マップとRGB画像から点群を生成.

    Args:
        depth_path: 深度マップ画像パス.
        image_path: 対応するRGB画像パス.
        output_path: 出力PLYパス.
        focal_length: カメラの焦点距離 (ピクセル単位).

    Returns:
        生成された点群ファイルのパス.
    """
    depth_img = np.array(Image.open(depth_path).convert("L")).astype(np.float32)
    rgb_img = np.array(Image.open(image_path).convert("RGB"))

    h, w = depth_img.shape
    cx, cy = w / 2.0, h / 2.0

    # ピクセル座標 → 3D座標への変換
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_img / 255.0 * 10.0  # スケーリング (仮)
    x = (u - cx) * z / focal_length
    y = (v - cy) * z / focal_length

    # 有効な点のみ抽出
    mask = z > 0.01
    points = np.stack([x[mask], y[mask], z[mask]], axis=-1)
    colors = rgb_img[mask]

    # PLY形式で出力
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

    logger.info("Point cloud generated: %s (%d points)", output_path, len(points))
    return output_path
