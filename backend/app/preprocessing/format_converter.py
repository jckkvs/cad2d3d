"""
ファイルフォーマット変換モジュール.

PDF → 画像, DXF → 画像, HEIC → 画像 等の変換処理.

Implements: F-040 (多形式入力対応)
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from app.core.exceptions import PreprocessingError

logger = logging.getLogger(__name__)

# 対応フォーマットマッピング
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_VECTOR_EXTENSIONS = {".svg", ".dxf", ".dwg"}
_DOCUMENT_EXTENSIONS = {".pdf"}
_RAW_EXTENSIONS = {".heic", ".heif"}


def convert_to_images(input_path: Path) -> list[Path]:
    """
    入力ファイルをPNG画像に変換.

    Args:
        input_path: 入力ファイルパス.

    Returns:
        変換後の画像パスのリスト (1入力→複数ページの場合あり).

    Raises:
        PreprocessingError: 変換失敗.
    """
    ext = input_path.suffix.lower()
    output_dir = input_path.parent / f"{input_path.stem}_converted"
    output_dir.mkdir(parents=True, exist_ok=True)

    if ext in _IMAGE_EXTENSIONS:
        # 画像: そのままPNGに正規化
        return [_normalize_image(input_path, output_dir)]

    elif ext in _RAW_EXTENSIONS:
        return [_convert_heic(input_path, output_dir)]

    elif ext == ".pdf":
        return _convert_pdf(input_path, output_dir)

    elif ext == ".svg":
        return [_convert_svg(input_path, output_dir)]

    elif ext in {".dxf", ".dwg"}:
        return [_convert_cad_drawing(input_path, output_dir)]

    else:
        raise PreprocessingError(
            f"未対応のファイル形式: {ext}",
            detail="対応形式: " + ", ".join(sorted(_IMAGE_EXTENSIONS | _VECTOR_EXTENSIONS | _DOCUMENT_EXTENSIONS | _RAW_EXTENSIONS)),
        )


def _normalize_image(path: Path, output_dir: Path) -> Path:
    """画像をPNGに正規化."""
    try:
        img = Image.open(path)
        if img.mode in ("RGBA", "LA"):
            # 透明→白背景に合成
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img, mask=img.split()[1])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        out_path = output_dir / f"{path.stem}.png"
        img.save(out_path, "PNG")
        logger.info("Normalized image: %s -> %s", path, out_path)
        return out_path
    except Exception as e:
        raise PreprocessingError(f"画像の正規化に失敗: {e}") from e


def _convert_heic(path: Path, output_dir: Path) -> Path:
    """HEIC/HEIF → PNG 変換."""
    try:
        # pillow-heif がインストールされていれば利用
        import pillow_heif
        heif_file = pillow_heif.read_heif(str(path))
        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
        out_path = output_dir / f"{path.stem}.png"
        img.save(out_path, "PNG")
        return out_path
    except ImportError:
        # ffmpegフォールバック
        try:
            out_path = output_dir / f"{path.stem}.png"
            subprocess.run(
                ["ffmpeg", "-i", str(path), "-y", str(out_path)],
                check=True, capture_output=True,
            )
            return out_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise PreprocessingError(
                f"HEIC変換失敗: pillow-heif か ffmpeg が必要です: {e}",
                detail="pip install pillow-heif",
            ) from e


def _convert_pdf(path: Path, output_dir: Path) -> list[Path]:
    """PDF → PNG 変換 (全ページ)."""
    results: list[Path] = []
    try:
        # pdf2image がインストールされていれば利用
        from pdf2image import convert_from_path
        images = convert_from_path(str(path), dpi=300)
        for i, img in enumerate(images):
            out_path = output_dir / f"{path.stem}_page{i+1:03d}.png"
            img.save(out_path, "PNG")
            results.append(out_path)
            logger.info("PDF page %d -> %s", i + 1, out_path)
        return results
    except ImportError:
        # PyMuPDFフォールバック
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            for i, page in enumerate(doc):
                mat = fitz.Matrix(3.0, 3.0)  # 300 DPI相当
                pix = page.get_pixmap(matrix=mat)
                out_path = output_dir / f"{path.stem}_page{i+1:03d}.png"
                pix.save(str(out_path))
                results.append(out_path)
            doc.close()
            return results
        except ImportError as e:
            raise PreprocessingError(
                "PDF変換には pdf2image または PyMuPDF が必要です",
                detail="pip install pdf2image  または pip install pymupdf",
            ) from e


def _convert_svg(path: Path, output_dir: Path) -> Path:
    """SVG → PNG 変換."""
    try:
        # cairosvg がインストールされていれば利用
        import cairosvg
        out_path = output_dir / f"{path.stem}.png"
        cairosvg.svg2png(
            url=str(path),
            write_to=str(out_path),
            output_width=2048,
        )
        return out_path
    except ImportError:
        try:
            # Inkscape フォールバック
            out_path = output_dir / f"{path.stem}.png"
            subprocess.run(
                ["inkscape", str(path), "--export-type=png", f"--export-filename={out_path}", "-w", "2048"],
                check=True, capture_output=True,
            )
            return out_path
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise PreprocessingError(
                f"SVG変換失敗: cairosvg か Inkscape が必要です: {e}",
                detail="pip install cairosvg",
            ) from e


def _convert_cad_drawing(path: Path, output_dir: Path) -> Path:
    """DXF/DWG → PNG 変換."""
    ext = path.suffix.lower()
    try:
        if ext == ".dxf":
            # ezdxf でDXFを解析し matplotlibで描画
            import ezdxf
            from ezdxf.addons.drawing import Frontend, RenderContext
            from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
            import matplotlib.pyplot as plt

            doc = ezdxf.readfile(str(path))
            msp = doc.modelspace()
            fig = plt.figure(figsize=(20, 15), dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=True)
            ax.set_facecolor("white")
            out_path = output_dir / f"{path.stem}.png"
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return out_path
        else:
            # DWG: 外部ツール (LibreDWG/ODAFileConverter) にフォールバック
            raise PreprocessingError(
                "DWG形式はまだサポート準備中です",
                detail="DXF形式に変換してからアップロードしてください",
            )
    except ImportError as e:
        raise PreprocessingError(
            f"DXF変換には ezdxf + matplotlib が必要です: {e}",
            detail="pip install ezdxf matplotlib",
        ) from e
