"""
2D画像類似度比較モジュール.

画像間の視覚的類似度を比較し、過去の見積図面を効率的に参照する.

手法:
1. 特徴量抽出: ResNet / DINOv2 による埋め込みベクトル
2. ヒストグラム比較: 色・エッジヒストグラムのL2距離
3. 構造的類似性 (SSIM): 画像構造の類似度

Implements: F-081 (2D画像類似度)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.exceptions import CAD3DError

logger = logging.getLogger(__name__)


@dataclass
class ImageDescriptor:
    """画像の特徴量記述子."""
    edge_histogram: np.ndarray       # エッジ方向ヒストグラム (36bins)
    intensity_histogram: np.ndarray  # 輝度ヒストグラム (64bins)
    aspect_ratio: float
    mean_intensity: float
    edge_density: float
    source_path: str = ""
    embedding: np.ndarray | None = None  # DNN埋め込み (384-dim)


@dataclass
class ImageSimilarityResult:
    """画像類似度比較の結果."""
    score: float                # 総合スコア (0-1)
    histogram_similarity: float
    edge_similarity: float
    aspect_similarity: float
    embedding_similarity: float | None = None


def compute_image_descriptor(
    image_path: Path,
    use_dnn: bool = False,
) -> ImageDescriptor:
    """
    画像の特徴量記述子を計算.

    Args:
        image_path: 画像ファイルパス.
        use_dnn: DNN埋め込みを使用するか (要torch + transformers).
    """
    try:
        img = Image.open(image_path).convert("L")  # グレースケール
    except Exception as e:
        raise CAD3DError(f"画像の読み込みに失敗: {e}")

    arr = np.array(img, dtype=np.float32) / 255.0

    # 輝度ヒストグラム
    intensity_hist, _ = np.histogram(arr.flatten(), bins=64, range=(0, 1), density=True)

    # エッジ方向ヒストグラム (Sobel)
    edge_hist, edge_density = _compute_edge_histogram(arr)

    # アスペクト比
    h, w = arr.shape
    aspect_ratio = w / h if h > 0 else 1.0

    # 平均輝度
    mean_intensity = float(arr.mean())

    # DNN埋め込み
    embedding = None
    if use_dnn:
        try:
            embedding = _compute_dnn_embedding(image_path)
        except Exception:
            logger.warning("DNN embedding failed, using histogram only")

    return ImageDescriptor(
        edge_histogram=edge_hist,
        intensity_histogram=intensity_hist,
        aspect_ratio=aspect_ratio,
        mean_intensity=mean_intensity,
        edge_density=edge_density,
        source_path=str(image_path),
        embedding=embedding,
    )


def compare_images(
    desc_a: ImageDescriptor,
    desc_b: ImageDescriptor,
) -> ImageSimilarityResult:
    """2つの画像の類似度を比較."""
    # ヒストグラム類似度 (Bhattacharyya距離の逆数)
    hist_sim = _histogram_similarity(desc_a.intensity_histogram, desc_b.intensity_histogram)

    # エッジ方向類似度
    edge_sim = _histogram_similarity(desc_a.edge_histogram, desc_b.edge_histogram)

    # アスペクト比の類似度
    max_aspect = max(desc_a.aspect_ratio, desc_b.aspect_ratio)
    aspect_sim = min(desc_a.aspect_ratio, desc_b.aspect_ratio) / max_aspect if max_aspect > 0 else 1.0

    # DNN埋め込みの類似度
    emb_sim = None
    if desc_a.embedding is not None and desc_b.embedding is not None:
        dot = np.dot(desc_a.embedding, desc_b.embedding)
        na = np.linalg.norm(desc_a.embedding)
        nb = np.linalg.norm(desc_b.embedding)
        emb_sim = float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0

    # 総合スコア
    if emb_sim is not None:
        score = 0.5 * emb_sim + 0.2 * hist_sim + 0.2 * edge_sim + 0.1 * aspect_sim
    else:
        score = 0.35 * hist_sim + 0.45 * edge_sim + 0.2 * aspect_sim

    return ImageSimilarityResult(
        score=float(np.clip(score, 0, 1)),
        histogram_similarity=float(hist_sim),
        edge_similarity=float(edge_sim),
        aspect_similarity=float(aspect_sim),
        embedding_similarity=emb_sim,
    )


def compare_image_files(path_a: Path, path_b: Path, use_dnn: bool = False) -> ImageSimilarityResult:
    """ファイルパスから直接比較."""
    desc_a = compute_image_descriptor(path_a, use_dnn)
    desc_b = compute_image_descriptor(path_b, use_dnn)
    return compare_images(desc_a, desc_b)


def _compute_edge_histogram(arr: np.ndarray, bins: int = 36) -> tuple[np.ndarray, float]:
    """Sobelフィルタでエッジ方向ヒストグラムを計算."""
    # Sobel x, y
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    from scipy.ndimage import convolve
    gx = convolve(arr, kernel_x)
    gy = convolve(arr, kernel_y)

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)  # -π to π

    # エッジ閾値
    threshold = magnitude.mean() + magnitude.std()
    edge_mask = magnitude > threshold
    edge_density = float(edge_mask.sum() / edge_mask.size)

    # エッジ方向のヒストグラム
    if edge_mask.any():
        edge_dirs = direction[edge_mask]
        hist, _ = np.histogram(edge_dirs, bins=bins, range=(-np.pi, np.pi), density=True)
    else:
        hist = np.zeros(bins)

    return hist, edge_density


def _histogram_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya係数による類似度."""
    a_norm = a / (a.sum() + 1e-10)
    b_norm = b / (b.sum() + 1e-10)
    bc = np.sum(np.sqrt(a_norm * b_norm))
    return float(np.clip(bc, 0, 1))


def _compute_dnn_embedding(image_path: Path) -> np.ndarray:
    """DINOv2/ResNetで画像埋め込みを計算."""
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        model_name = "facebook/dinov2-small"
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        img = Image.open(image_path).convert("RGB")
        inputs = extractor(images=img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        return embedding
    except ImportError:
        raise CAD3DError("DNN embedding requires torch + transformers")
