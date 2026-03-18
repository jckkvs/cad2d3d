"""
CAD3D Generator — NiceGUI フロントエンド.

Node.js 不要で動作する Python ベースの Web UI.
ui.scene (Three.js) で 3D ビューアを内蔵.
FastAPI バックエンドの API を直接呼び出す.

起動方法: python nicegui_app.py
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from nicegui import app, ui, events

# バックエンドモジュールの再利用
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.engines.registry import EngineRegistry
from app.engines.base import ProcessedImage, ReconstructionParams
from app.models.schemas import EngineInfo, EngineStatus, OutputFormat
from app.weights.manager import weight_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── グローバル状態 ─────────────────────────────────────

uploaded_files: list[dict[str, Any]] = []
current_engine: str = ""
current_job_id: str | None = None
generation_result_path: Path | None = None


# ── ヘルパー関数 ──────────────────────────────────────

def format_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def get_engines() -> list[EngineInfo]:
    try:
        return EngineRegistry.list_available()
    except Exception:
        return []


# ── カスタム CSS ──────────────────────────────────────

CUSTOM_CSS = """
<style>
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-tertiary: #1e293b;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.3);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-tertiary: #64748b;
    --border: rgba(148, 163, 184, 0.15);
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
}
body {
    background: var(--bg-primary) !important;
    background-image:
        radial-gradient(ellipse at 20% 0%, rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 100%, rgba(139,92,246,0.06) 0%, transparent 60%) !important;
}
.q-page { background: transparent !important; }
.q-card { background: var(--bg-secondary) !important; border: 1px solid var(--border) !important; }
.q-toolbar { background: rgba(17,24,39,0.7) !important; backdrop-filter: blur(12px); }
.nicegui-content { padding: 0 !important; }
.file-chip { 
    background: var(--bg-tertiary) !important; 
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
.engine-card {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    transition: all 0.25s ease !important;
    cursor: pointer;
}
.engine-card:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 15px var(--accent-glow) !important;
}
.engine-card.selected {
    border-color: var(--accent) !important;
    background: rgba(99, 102, 241, 0.08) !important;
}
.cap-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(99, 102, 241, 0.1);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.2);
    display: inline-block;
    margin: 2px;
}
.gradient-text {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.status-ready { color: #10b981; }
.status-missing { color: #f59e0b; }
.status-error { color: #ef4444; }
</style>
"""


# ── メインページ ──────────────────────────────────────

@ui.page("/")
async def main_page():
    global current_engine

    # 初期化
    settings.ensure_dirs()
    EngineRegistry.discover_engines()
    engines = get_engines()
    if engines and not current_engine:
        current_engine = engines[0].name

    ui.add_head_html(CUSTOM_CSS)
    ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Noto+Sans+JP:wght@300;400;500;600;700&display=swap" rel="stylesheet">')

    # ── ヘッダー ──────────────────────────────────

    with ui.header(elevated=True).classes("items-center q-px-lg").style(
        "background: rgba(17,24,39,0.85); backdrop-filter: blur(12px); border-bottom: 1px solid rgba(148,163,184,0.15);"
    ):
        ui.label("◆").style("font-size: 1.5rem; margin-right: 8px;")
        ui.label("CAD3D Generator").classes("gradient-text").style(
            "font-size: 1.3rem; font-weight: 700; font-family: Inter, 'Noto Sans JP', sans-serif;"
        )
        ui.label("v0.1.0").style(
            "font-size: 0.7rem; color: #64748b; padding: 2px 6px; border: 1px solid rgba(148,163,184,0.15); border-radius: 4px; margin-left: 8px;"
        )
        ui.space()
        ui.label("NiceGUI Edition").style("color: #64748b; font-size: 0.75rem;")
        settings_btn = ui.button(icon="settings", on_click=lambda: settings_dialog.open()).props(
            "flat round color=grey-6"
        )

    # ── 結果操作ヘルパー関数 (UI構築前に定義) ────────

    def load_model_to_scene(scene_widget, model_path: Path):
        """3Dシーンにモデルをロード."""
        try:
            import trimesh
            mesh = trimesh.load(str(model_path))

            scene_widget.clear()
            with scene_widget:
                if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                    verts = mesh.vertices
                    center = verts.mean(axis=0)
                    scale = 2.0 / max(verts.max(axis=0) - verts.min(axis=0))

                    extent = mesh.bounding_box.extents
                    ui.scene.box(extent[0] * scale, extent[1] * scale, extent[2] * scale).material(
                        color="#6366f1", opacity=0.8
                    )

            logger.info("Loaded model to 3D scene: %s", model_path)
        except Exception:
            logger.warning("Failed to load model to scene", exc_info=True)

    def download_result():
        """結果ファイルをダウンロード."""
        if generation_result_path and generation_result_path.exists():
            ui.download(str(generation_result_path))
        else:
            ui.notify("出力ファイルが見つかりません", type="warning")

    async def open_external():
        """外部CADソフトで開く."""
        import os, subprocess, sys as _sys
        if not generation_result_path or not generation_result_path.exists():
            ui.notify("出力ファイルが見つかりません", type="warning")
            return
        try:
            if _sys.platform == "win32":
                os.startfile(str(generation_result_path))
            elif _sys.platform == "darwin":
                subprocess.run(["open", str(generation_result_path)], check=True)
            else:
                subprocess.run(["xdg-open", str(generation_result_path)], check=True)
            ui.notify("✓ 外部アプリで開きました", type="positive")
        except Exception as e:
            ui.notify(f"✕ 起動失敗: {e}", type="negative")

    async def reload_model():
        """外部編集後のモデルを再読み込み."""
        if generation_result_path and generation_result_path.exists():
            try:
                load_model_to_scene(scene, generation_result_path)
                ui.notify("✓ モデルを再読み込みしました", type="positive")
            except Exception as e:
                ui.notify(f"✕ 再読み込み失敗: {e}", type="negative")
        else:
            ui.notify("出力ファイルが見つかりません", type="warning")

    # ── メインレイアウト ──────────────────────────

    with ui.row().classes("w-full").style("height: calc(100vh - 64px); gap: 0;"):

        # ===== 左ペイン：入力・エンジン =====
        with ui.column().style(
            "width: 400px; min-width: 400px; background: #111827; border-right: 1px solid rgba(148,163,184,0.15); overflow-y: auto;"
        ):

            # タブ
            with ui.tabs().classes("w-full").style("background: #0a0e1a;") as tabs:
                tab_input = ui.tab("入力", icon="upload_file")
                tab_engine = ui.tab("エンジン", icon="settings_suggest")
                tab_compare = ui.tab("比較", icon="compare")
                tab_weights = ui.tab("重み", icon="inventory_2")
                tab_mold = ui.tab("金型", icon="precision_manufacturing")

            with ui.tab_panels(tabs, value=tab_input).classes("w-full").style(
                "background: transparent;"
            ):
                # ── タブ: 入力 ─────────────────

                with ui.tab_panel(tab_input):
                    ui.label("📤 ファイル入力").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )

                    # アップロード領域
                    upload_el = ui.upload(
                        label="クリックまたはドラッグ＆ドロップで画像/CADファイルを追加\nJPG, PNG, PDF, DXF, SVG, TIFF, HEIC 等",
                        multiple=True,
                        auto_upload=True,
                        on_upload=lambda e: handle_upload(e),
                    ).props('accept=".jpg,.jpeg,.png,.bmp,.tiff,.tif,.pdf,.dxf,.svg,.heic,.webp,.stl,.obj,.glb,.ply" color=indigo-10').classes(
                        "w-full"
                    ).style("border: 2px dashed rgba(148,163,184,0.15); border-radius: 16px; min-height: 120px;")

                    # ファイル一覧
                    file_list_container = ui.column().classes("w-full gap-1 q-mt-sm")

                    ui.separator().style("background: rgba(148,163,184,0.15);")

                    # 出力設定
                    ui.label("📐 出力設定").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )
                    output_format = ui.select(
                        options={
                            "glb": "GLB (Web3D)",
                            "obj": "OBJ (メッシュ)",
                            "stl": "STL (3Dプリント)",
                            "ply": "PLY (点群)",
                            "gltf": "glTF (JSON)",
                            "step": "STEP (CAD)",
                        },
                        value="glb",
                        label="出力フォーマット",
                    ).classes("w-full").props("outlined dark color=indigo-4")

                    ui.separator().style("background: rgba(148,163,184,0.15);")

                    # 生成ボタン
                    generate_btn = ui.button(
                        "🚀 3Dモデルを生成",
                        on_click=lambda: run_generation(),
                    ).classes("w-full").props("color=indigo size=lg unelevated").style(
                        "background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; font-weight: 600; border-radius: 12px;"
                    )

                    # 進捗
                    progress_container = ui.column().classes("w-full q-mt-sm").style("display: none;")
                    with progress_container:
                        progress_label = ui.label("").style("color: #94a3b8; font-size: 0.85rem;")
                        progress_bar = ui.linear_progress(value=0, show_value=False).props("color=indigo rounded")

                # ── タブ: エンジン ─────────────

                with ui.tab_panel(tab_engine):
                    ui.label("🔧 生成エンジン選択").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )

                    engine_container = ui.column().classes("w-full gap-2 q-mt-sm")

                    for eng in engines:
                        with ui.card().classes(
                            f"w-full engine-card {'selected' if eng.name == current_engine else ''}"
                        ).on("click", lambda e=eng: select_engine(e.name, engine_container)):
                            with ui.row().classes("items-center justify-between w-full"):
                                ui.label(eng.display_name).style("font-weight: 600; font-size: 0.9rem; color: #f1f5f9;")
                                status_cls = {
                                    "ready": "status-ready",
                                    "weights_missing": "status-missing",
                                }.get(eng.status, "status-error")
                                status_text = {
                                    "ready": "✓ Ready",
                                    "weights_missing": "⚠ 重み不足",
                                    "dependency_missing": "✕ 依存関係不足",
                                }.get(eng.status, "✕ Error")
                                ui.label(status_text).classes(status_cls).style("font-size: 0.7rem; font-weight: 500;")

                            ui.label(eng.description).style("font-size: 0.75rem; color: #94a3b8; line-height: 1.5;")

                            with ui.row().classes("gap-1 q-mt-xs"):
                                if eng.capabilities.supports_single_image:
                                    ui.html('<span class="cap-badge">1画像</span>')
                                if eng.capabilities.supports_multi_image:
                                    ui.html('<span class="cap-badge">複数画像</span>')
                                if eng.capabilities.outputs_mesh:
                                    ui.html('<span class="cap-badge">メッシュ</span>')
                                if eng.capabilities.outputs_cad:
                                    ui.html('<span class="cap-badge">CAD</span>')
                                if eng.capabilities.requires_gpu:
                                    ui.html('<span class="cap-badge">GPU</span>')
                                if eng.capabilities.estimated_vram_gb:
                                    ui.html(f'<span class="cap-badge">VRAM {eng.capabilities.estimated_vram_gb}GB</span>')

                            with ui.row().classes("q-mt-xs gap-1"):
                                ui.button("📖 マニュアル", on_click=lambda e=eng: show_readme(e.name)).props(
                                    "flat dense color=indigo-4 size=sm"
                                )

                # ── タブ: 重み ─────────────────

                with ui.tab_panel(tab_weights):
                    ui.label("📦 モデル重み管理").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )

                    weights_container = ui.column().classes("w-full q-mt-sm")
                    # refresh_weights_ui は後で定義されるため、タイマーで遅延呼び出し
                    ui.timer(0.1, lambda: refresh_weights_ui(weights_container), once=True)

                # ── タブ: 金型設計 ─────────────

                with ui.tab_panel(tab_mold):
                    ui.label("🏭 金型設計推定").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )
                    ui.label("3Dメッシュ (STL/OBJ/GLB) をアップロードして金型構造を推定").style(
                        "font-size: 0.75rem; color: #94a3b8;"
                    )

                    ui.separator().style("background: rgba(148,163,184,0.15);")

                    # 金型解析設定
                    mold_cavity_count = ui.select(
                        options={"1": "1個取り", "2": "2個取り", "4": "4個取り"},
                        value="1", label="キャビティ数",
                    ).classes("w-full").props("outlined dark color=indigo-4")

                    # 解析ボタン群
                    mold_result_container = ui.column().classes("w-full gap-2 q-mt-sm")

                    ui.button(
                        "🔍 アンダーカット解析",
                        on_click=lambda: run_undercut_analysis(mold_result_container),
                    ).classes("w-full").props("color=amber-8 unelevated").style("border-radius: 12px;")

                    ui.button(
                        "📐 パーティングライン最適化",
                        on_click=lambda: run_pl_optimization(mold_result_container),
                    ).classes("w-full").props("color=teal-8 unelevated").style("border-radius: 12px;")

                    ui.button(
                        "📏 ドラフト角解析",
                        on_click=lambda: run_draft_analysis(mold_result_container),
                    ).classes("w-full").props("color=cyan-8 unelevated").style("border-radius: 12px;")

                    ui.button(
                        "🏗️ 金型全体設計",
                        on_click=lambda: run_full_mold_design(mold_result_container),
                    ).classes("w-full").props("color=indigo unelevated").style(
                        "background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; font-weight: 600; border-radius: 12px;"
                    )

                    ui.separator().style("background: rgba(148,163,184,0.15);")

                    # 類似度比較
                    ui.label("🔍 類似度比較").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )
                    ui.label("2つのファイルをアップロードして類似度を比較").style(
                        "font-size: 0.7rem; color: #94a3b8;"
                    )
                    ui.button(
                        "📊 類似度を計算",
                        on_click=lambda: run_similarity(mold_result_container),
                    ).classes("w-full").props("flat color=indigo-4")

                    ui.separator().style("background: rgba(148,163,184,0.15);")

                    # 部品DB
                    ui.label("📦 標準部品DB").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )
                    parts_category_select = ui.select(
                        options={
                            "guide_pin": "ガイドピン",
                            "ejector_pin": "突き出しピン",
                            "gas_spring": "ガススプリング",
                            "sprue_bushing": "スプルーブッシュ",
                        },
                        value="guide_pin", label="カテゴリ",
                    ).classes("w-full").props("outlined dark color=indigo-4")
                    ui.button(
                        "📋 部品一覧を表示",
                        on_click=lambda: show_parts_list(parts_category_select.value, mold_result_container),
                    ).classes("w-full").props("flat color=indigo-4")

                    ui.separator().style("background: rgba(148,163,184,0.15);")

                    # サイジング
                    ui.label("⚙ サイジング計算").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )
                    sizing_resin = ui.select(
                        options={
                            "PP": "PP (ポリプロピレン)", "PE": "PE", "ABS": "ABS",
                            "PA": "PA (ナイロン)", "POM": "POM", "PC": "PC",
                            "PBT": "PBT", "PS": "PS", "PMMA": "PMMA (アクリル)", "PPS": "PPS",
                        },
                        value="ABS", label="樹脂材料",
                    ).classes("w-full").props("outlined dark color=indigo-4")
                    sizing_area = ui.number(label="投影面積 [mm²]", value=5000, min=100, max=500000).classes("w-full").props("outlined dark color=indigo-4")
                    sizing_thickness = ui.number(label="肉厚 [mm]", value=2.5, min=0.3, max=20, step=0.1).classes("w-full").props("outlined dark color=indigo-4")
                    sizing_weight = ui.number(label="製品重量 [g]", value=30, min=0.1, max=50000).classes("w-full").props("outlined dark color=indigo-4")
                    sizing_qty = ui.number(label="生産数量", value=10000, min=1, max=10000000, step=1000).classes("w-full").props("outlined dark color=indigo-4")

                    ui.button(
                        "📊 サイジング計算",
                        on_click=lambda: run_sizing(mold_result_container),
                    ).classes("w-full").props("color=deep-purple unelevated").style("border-radius: 12px;")

                # ── タブ: 比較 ─────────────────

                with ui.tab_panel(tab_compare):
                    ui.label("🔄 エンジン比較").style(
                        "font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;"
                    )
                    ui.label("複数エンジンで同時に3D変換し、結果を比較します").style(
                        "font-size: 0.7rem; color: #94a3b8;"
                    )

                    ui.separator().style("background: rgba(148,163,184,0.15);")

                    # エンジン選択チェックボックス群
                    ui.label("📋 比較するエンジンを選択").style(
                        "font-size: 0.75rem; font-weight: 600; color: #a5b4fc; margin-top: 8px;"
                    )

                    compare_selection: dict[str, ui.checkbox] = {}
                    compare_engines_container = ui.column().classes("w-full gap-1 q-mt-xs")

                    for eng in engines:
                        with compare_engines_container:
                            with ui.row().classes("items-center gap-2").style(
                                "padding: 6px 12px; border-radius: 8px; background: rgba(30,41,59,0.6);"
                            ):
                                cb = ui.checkbox(eng.display_name, value=False).props("dense color=indigo-4")
                                compare_selection[eng.name] = cb
                                status_text = "✓" if eng.status == "ready" else "⚠"
                                st_color = "#10b981" if eng.status == "ready" else "#f59e0b"
                                ui.label(status_text).style(f"color: {st_color}; font-size: 0.7rem;")
                                if eng.capabilities.estimated_vram_gb:
                                    ui.label(f"{eng.capabilities.estimated_vram_gb}GB").style(
                                        "font-size: 0.65rem; color: #64748b;"
                                    )

                    ui.separator().style("background: rgba(148,163,184,0.15);")

                    # 比較実行ボタン
                    compare_btn = ui.button(
                        "⚡ 比較生成を開始",
                        on_click=lambda: run_comparison_generation(
                            compare_selection, compare_result_container
                        ),
                    ).classes("w-full").props("color=indigo unelevated").style(
                        "background: linear-gradient(135deg, #6366f1, #a855f7) !important; "
                        "font-weight: 600; border-radius: 12px;"
                    )

                    # 比較結果エリア
                    compare_result_container = ui.column().classes("w-full gap-2 q-mt-sm")

        # ===== 右ペイン：3Dビューア =====
        with ui.column().classes("col").style("background: #0a0e1a; position: relative;"):

            # ビューアツールバー
            with ui.row().classes("w-full items-center q-pa-sm").style(
                "background: rgba(17,24,39,0.7); backdrop-filter: blur(12px); border-bottom: 1px solid rgba(148,163,184,0.15);"
            ):
                ui.label("3D プレビュー").style("color: #94a3b8; font-size: 0.8rem;")
                ui.space()
                # 結果操作ボタン（生成完了後に表示）
                result_buttons = ui.row().classes("gap-1").style("display: none;")
                with result_buttons:
                    ui.button("💾 ダウンロード", on_click=lambda: download_result()).props("flat dense color=grey-4 size=sm")
                    ui.button("🔗 外部CADで開く", on_click=lambda: open_external()).props("flat dense color=grey-4 size=sm")
                    ui.button("🔄 再読み込み", on_click=lambda: reload_model()).props("flat dense color=grey-4 size=sm")

            # 3Dシーン
            scene_container = ui.column().classes("w-full col items-center justify-center")
            with scene_container:
                # Three.js 3Dシーン
                scene = ui.scene(width=800, height=600).style(
                    "background: #0a0e1a !important; border-radius: 8px;"
                )
                with scene:
                    # 初期グリッド表示
                    pass

                # 空状態メッセージ
                empty_msg = ui.column().classes("items-center gap-2 q-mt-lg")
                with empty_msg:
                    ui.label("◇").style("font-size: 3rem; color: #64748b; opacity: 0.3;")
                    ui.label("3D ビューア").style("font-size: 1rem; font-weight: 500; color: #64748b;")
                    ui.label(
                        "左パネルから画像をアップロードし、エンジンを選択して\n「3Dモデルを生成」を押してください。"
                    ).style("font-size: 0.8rem; color: #475569; text-align: center; white-space: pre-line;")

    # ── 設定ダイアログ ────────────────────────────

    settings_dialog = ui.dialog()
    with settings_dialog, ui.card().style(
        "background: #111827; border: 1px solid rgba(148,163,184,0.15); min-width: 500px; border-radius: 16px;"
    ):
        with ui.row().classes("items-center justify-between w-full q-mb-md"):
            ui.label("⚙ 高度設定").style("font-size: 1.1rem; font-weight: 600; color: #f1f5f9;")
            ui.button(icon="close", on_click=settings_dialog.close).props("flat round color=grey-6")

        proxy_http = ui.input(label="HTTP プロキシ", placeholder="http://proxy.example.com:8080").classes(
            "w-full"
        ).props("outlined dark color=indigo-4")
        proxy_https = ui.input(label="HTTPS プロキシ", placeholder="http://proxy.example.com:8080").classes(
            "w-full"
        ).props("outlined dark color=indigo-4")
        hf_token = ui.input(label="HuggingFace Token", placeholder="hf_xxxxxxxxxxxx", password=True).classes(
            "w-full"
        ).props("outlined dark color=indigo-4")
        ui.link("HuggingFace Token を作成", "https://huggingface.co/settings/tokens", new_tab=True).style(
            "color: #a5b4fc; font-size: 0.7rem;"
        )

        with ui.row().classes("w-full justify-end q-mt-md gap-2"):
            ui.button("キャンセル", on_click=settings_dialog.close).props("flat color=grey-6")
            ui.button("保存", on_click=lambda: save_settings(
                proxy_http.value, proxy_https.value, hf_token.value, settings_dialog
            )).props("color=indigo unelevated")

    # README ダイアログ
    readme_dialog = ui.dialog()
    readme_content_label = ui.label()
    with readme_dialog, ui.card().style(
        "background: #111827; border: 1px solid rgba(148,163,184,0.15); min-width: 600px; max-height: 80vh; overflow-y: auto; border-radius: 16px;"
    ):
        with ui.row().classes("items-center justify-between w-full q-mb-md"):
            ui.label("📖 エンジンマニュアル").style("font-size: 1.1rem; font-weight: 600; color: #f1f5f9;")
            ui.button(icon="close", on_click=readme_dialog.close).props("flat round color=grey-6")
        readme_content_el = ui.markdown("").style("color: #94a3b8; font-size: 0.85rem;")

    # ── イベントハンドラ ──────────────────────────

    async def handle_upload(e: events.UploadEventArguments):
        """ファイルアップロード処理."""
        file_id = str(uuid.uuid4())
        ext = Path(e.name).suffix.lower()
        stored_name = f"{file_id}{ext}"
        stored_path = settings.upload_dir / stored_name
        stored_path.write_bytes(e.content.read())

        file_info = {
            "id": file_id,
            "name": e.name,
            "size": stored_path.stat().st_size,
            "path": str(stored_path),
            "view_angle": "unknown",
        }
        uploaded_files.append(file_info)
        refresh_file_list(file_list_container)
        ui.notify(f"✓ {e.name} をアップロードしました", type="positive")

    def refresh_file_list(container):
        """ファイル一覧UIを更新."""
        container.clear()
        with container:
            for f in uploaded_files:
                with ui.row().classes("w-full items-center gap-2 q-pa-xs").style(
                    "background: #1e293b; border-radius: 8px; border: 1px solid rgba(148,163,184,0.15);"
                ):
                    ui.icon("description", color="indigo-4").style("font-size: 1rem;")
                    ui.label(f["name"]).style(
                        "flex: 1; font-size: 0.8rem; color: #f1f5f9; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"
                    )
                    ui.label(format_size(f["size"])).style("font-size: 0.7rem; color: #64748b;")
                    ui.select(
                        options={
                            "unknown": "自動",
                            "front": "正面",
                            "back": "背面",
                            "left": "左側",
                            "right": "右側",
                            "top": "上面",
                            "bottom": "底面",
                            "isometric": "等角",
                        },
                        value=f["view_angle"],
                        on_change=lambda e, fid=f["id"]: set_view_angle(fid, e.value),
                    ).props("dense outlined dark color=indigo-4").style("width: 80px; font-size: 0.7rem;")
                    ui.button(icon="close", on_click=lambda fid=f["id"]: remove_file(fid, container)).props(
                        "flat round dense color=grey-6 size=xs"
                    )

    def set_view_angle(file_id: str, angle: str):
        for f in uploaded_files:
            if f["id"] == file_id:
                f["view_angle"] = angle
                break

    def remove_file(file_id: str, container):
        global uploaded_files
        for f in uploaded_files:
            if f["id"] == file_id:
                Path(f["path"]).unlink(missing_ok=True)
                break
        uploaded_files = [f for f in uploaded_files if f["id"] != file_id]
        refresh_file_list(container)

    def select_engine(name: str, container):
        global current_engine
        current_engine = name
        ui.notify(f"エンジンを {name} に変更しました", type="info")
        # ページをリロードしてUI更新
        ui.navigate.to("/")

    def refresh_weights_ui(container):
        """重み管理UIを更新."""
        container.clear()
        with container:
            if not current_engine:
                ui.label("エンジンを選択してください。").style("color: #64748b; font-size: 0.85rem;")
                return

            try:
                engine = EngineRegistry.get(current_engine)
                info = engine.get_info()
                ui.label(f"{info.display_name} の重みファイル").style(
                    "color: #94a3b8; font-size: 0.8rem; font-weight: 500;"
                )

                for i, w in enumerate(info.required_weights):
                    exists = weight_manager.check_weight_exists(w)
                    with ui.row().classes("w-full items-center gap-2 q-pa-sm").style(
                        "background: #1e293b; border-radius: 8px; border: 1px solid rgba(148,163,184,0.15);"
                    ):
                        ui.icon("circle", color="green" if exists else "orange").style("font-size: 0.5rem;")
                        ui.label(w.name).style("flex: 1; font-size: 0.8rem; color: #f1f5f9;")
                        if not exists:
                            ui.button(
                                "ダウンロード",
                                on_click=lambda w=w: download_weight(w),
                            ).props("flat dense color=indigo-4 size=sm")
                        else:
                            ui.label("✓").style("color: #10b981;")

                ui.button("📥 全てダウンロード", on_click=lambda: download_all_weights()).classes(
                    "w-full q-mt-sm"
                ).props("color=indigo unelevated")

                ui.button("📖 手動配置ガイド", on_click=lambda: show_readme(current_engine)).classes(
                    "w-full q-mt-xs"
                ).props("flat color=indigo-4")

            except Exception as e:
                ui.label(f"エラー: {e}").style("color: #ef4444; font-size: 0.8rem;")

    async def download_weight(w):
        ui.notify("ダウンロード中...", type="info")
        try:
            await weight_manager.download_weight(w)
            ui.notify(f"✓ {w.name} のダウンロード完了", type="positive")
        except Exception as e:
            ui.notify(f"✕ ダウンロード失敗: {e}", type="negative")

    async def download_all_weights():
        try:
            engine = EngineRegistry.get(current_engine)
            for w in engine.get_required_weights():
                if not weight_manager.check_weight_exists(w):
                    await weight_manager.download_weight(w)
            ui.notify("✓ 全ての重みのダウンロード完了", type="positive")
        except Exception as e:
            ui.notify(f"✕ ダウンロード失敗: {e}", type="negative")

    async def show_readme(engine_name: str):
        try:
            engine = EngineRegistry.get(engine_name)
            info = engine.get_info()
            if info.readme_path and Path(info.readme_path).exists():
                content = Path(info.readme_path).read_text(encoding="utf-8")
                readme_content_el.set_content(content)
            else:
                readme_content_el.set_content("README_MODEL.md が見つかりません。")
            readme_dialog.open()
        except Exception as e:
            ui.notify(f"README の取得に失敗: {e}", type="negative")

    async def run_generation():
        """3D生成を実行."""
        global current_job_id, generation_result_path

        if not uploaded_files:
            ui.notify("⚠ 画像をアップロードしてください", type="warning")
            return
        if not current_engine:
            ui.notify("⚠ エンジンを選択してください", type="warning")
            return

        try:
            engine = EngineRegistry.get(current_engine)
            if engine.check_ready() != EngineStatus.READY:
                ui.notify("⚠ エンジンの準備ができていません。重みをダウンロードしてください。", type="warning")
                return
        except Exception as e:
            ui.notify(f"✕ エンジンエラー: {e}", type="negative")
            return

        # UI更新
        progress_container.style("display: block;")
        generate_btn.props("loading")
        progress_bar.set_value(0)
        progress_label.set_text("ジョブを開始します...")

        job_id = str(uuid.uuid4())
        current_job_id = job_id
        output_dir = settings.temp_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        def progress_cb(p: float, msg: str):
            progress_bar.set_value(p)
            progress_label.set_text(msg)

        images = [
            ProcessedImage(
                path=Path(f["path"]),
                view_angle=f["view_angle"],
            )
            for f in uploaded_files
        ]

        params = ReconstructionParams(
            output_format=OutputFormat(output_format.value),
            output_dir=output_dir,
        )

        try:
            result = await engine.reconstruct(images, params, progress_cb)

            if result.success and result.output_path:
                generation_result_path = result.output_path
                progress_bar.set_value(1.0)
                progress_label.set_text(f"✓ 生成完了! ({result.elapsed_seconds:.1f}秒)")
                ui.notify("🎉 3Dモデルの生成が完了しました！", type="positive")
                result_buttons.style("display: flex;")

                # 3Dシーンにモデルを表示 (STL/OBJの場合)
                try:
                    load_model_to_scene(scene, result.output_path)
                    empty_msg.style("display: none;")
                except Exception:
                    logger.warning("3Dシーンへのロード失敗", exc_info=True)
            else:
                progress_label.set_text(f"✕ 生成失敗: {result.error_message}")
                ui.notify(f"✕ 生成失敗: {result.error_message}", type="negative")

        except Exception as e:
            progress_label.set_text(f"✕ エラー: {e}")
            ui.notify(f"✕ エラー: {e}", type="negative")
            logger.error("Generation failed", exc_info=True)
        finally:
            generate_btn.props(remove="loading")

    # (download_result/open_external/reload_model/load_model_to_scene は
    #  UIビルド前に定義済み — L163付近を参照)

    # ── 金型設計イベントハンドラ ───────────────

    def _get_mesh_path() -> Path | None:
        """アップロード済みメッシュファイルを取得."""
        mesh_exts = {".stl", ".obj", ".glb", ".ply", ".gltf", ".step", ".stp"}
        for f in uploaded_files:
            ext = Path(f["path"]).suffix.lower()
            if ext in mesh_exts:
                return Path(f["path"])
        return None

    async def run_undercut_analysis(container):
        mesh_path = _get_mesh_path()
        if not mesh_path:
            ui.notify("⚠ 3Dメッシュ (STL/OBJ/GLB) をアップロードしてください", type="warning")
            return
        container.clear()
        with container:
            ui.label("⏳ アンダーカット解析中...").style("color: #94a3b8;")
        try:
            from app.mold.undercut import detect_undercuts
            result = detect_undercuts(mesh_path, ray_density=30)
            container.clear()
            with container:
                ui.label("✅ アンダーカット解析完了").style("color: #10b981; font-weight: 600;")
                if result.has_undercut:
                    ui.label(f"⚠ {result.undercut_count} 箇所のアンダーカット検出").style("color: #f59e0b; font-size: 0.85rem;")
                    ui.label(f"総面積: {result.total_undercut_area:.1f} mm²").style("color: #94a3b8; font-size: 0.8rem;")
                    for i, r in enumerate(result.undercut_regions):
                        with ui.card().style("background: #1e293b; border: 1px solid rgba(148,163,184,0.15); padding: 8px; border-radius: 8px;"):
                            ui.label(f"領域 #{i+1}").style("font-weight: 600; color: #f1f5f9; font-size: 0.8rem;")
                            ui.label(f"面積={r.area:.1f}mm², 深さ={r.depth:.1f}mm").style("color: #94a3b8; font-size: 0.75rem;")
                            severity_colors = {"minor": "#10b981", "moderate": "#f59e0b", "major": "#ef4444"}
                            ui.label(f"深刻度: {r.severity}").style(f"color: {severity_colors.get(r.severity, '#94a3b8')}; font-size: 0.75rem;")
                            ui.label(f"推奨: {r.recommended_mechanism}").style("color: #a5b4fc; font-size: 0.75rem;")
                else:
                    ui.label("✓ アンダーカットなし").style("color: #10b981; font-size: 0.85rem;")
                ui.label(f"Core面: {len(result.core_faces)}, Cavity面: {len(result.cavity_faces)}").style("color: #64748b; font-size: 0.75rem;")
        except Exception as e:
            container.clear()
            with container:
                ui.label(f"✕ エラー: {e}").style("color: #ef4444;")

    async def run_pl_optimization(container):
        mesh_path = _get_mesh_path()
        if not mesh_path:
            ui.notify("⚠ 3Dメッシュをアップロードしてください", type="warning")
            return
        container.clear()
        with container:
            ui.label("⏳ パーティングライン最適化中...").style("color: #94a3b8;")
        try:
            from app.mold.parting_line import optimize_parting_line
            result = optimize_parting_line(mesh_path)
            container.clear()
            with container:
                ui.label("✅ PL最適化完了").style("color: #10b981; font-weight: 600;")
                b = result.best
                ui.label(f"最適方向: [{b.direction[0]:.2f}, {b.direction[1]:.2f}, {b.direction[2]:.2f}]").style("color: #f1f5f9; font-size: 0.8rem;")
                ui.label(f"金型高さ: {b.mold_height:.1f}mm").style("color: #94a3b8; font-size: 0.8rem;")
                ui.label(f"ゲート配置: {b.gate_accessibility:.0%}, 加工容易度: {b.machinability:.0%}").style("color: #94a3b8; font-size: 0.8rem;")
                ui.label(f"候補 {len(result.candidates)} 方向を評価").style("color: #64748b; font-size: 0.75rem;")
        except Exception as e:
            container.clear()
            with container:
                ui.label(f"✕ エラー: {e}").style("color: #ef4444;")

    async def run_full_mold_design(container):
        mesh_path = _get_mesh_path()
        if not mesh_path:
            ui.notify("⚠ 3Dメッシュをアップロードしてください", type="warning")
            return
        container.clear()
        with container:
            ui.label("⏳ 金型全体設計中...").style("color: #94a3b8;")
        try:
            from app.mold.core import MoldDesigner
            cavity_count = int(mold_cavity_count.value) if mold_cavity_count.value else 1
            result = MoldDesigner().design(mesh_path, cavity_count=cavity_count)
            container.clear()
            with container:
                ui.label("✅ 金型設計完了").style("color: #10b981; font-weight: 600;")
                ui.label(f"金型サイズ: {result.total_width:.0f} × {result.total_height:.0f} × {result.total_depth:.0f} mm").style("color: #f1f5f9; font-size: 0.85rem;")
                ui.label(f"金型重量: {result.total_weight_kg:.1f} kg").style("color: #94a3b8; font-size: 0.8rem;")
                ui.label(f"構成部品数: {len(result.components)}").style("color: #94a3b8; font-size: 0.8rem;")
                ui.label(f"概算コスト: ¥{result.estimated_total_cost:,.0f}").style("color: #a5b4fc; font-weight: 600; font-size: 0.9rem;")

                # サイジング情報
                if result.recommended_machine:
                    with ui.card().style("background: #1e293b; border: 1px solid rgba(148,163,184,0.15); padding: 6px; border-radius: 8px;"):
                        ui.label(f"推奨成形機: {result.recommended_machine} ({result.clamp_force_ton:.0f}tf)").style("color: #a5b4fc; font-size: 0.8rem;")
                        ui.label(f"サイクル: {result.cycle_time_s:.1f}秒 ({result.shots_per_hour}ショット/h)").style("color: #94a3b8; font-size: 0.75rem;")
                        ui.label(f"鋼材: {result.steel_grade}").style("color: #94a3b8; font-size: 0.75rem;")

                # ドラフト角
                if result.draft_result:
                    dr = result.draft_result
                    color = "#10b981" if dr.compliance_ratio > 0.9 else "#f59e0b" if dr.compliance_ratio > 0.5 else "#ef4444"
                    ui.label(f"📏 ドラフト角合格率: {dr.compliance_ratio:.0%} (不足: {len(dr.insufficient_faces)}面)").style(f"color: {color}; font-size: 0.8rem;")

                # 部品一覧
                with ui.expansion("📦 部品一覧", icon="list").style("color: #94a3b8;"):
                    for c in result.components:
                        ui.label(f"• {c.name} ×{c.quantity} {c.size}").style("color: #94a3b8; font-size: 0.75rem;")

                # アンダーカット
                if result.undercut_result and result.undercut_result.has_undercut:
                    ui.label(f"⚠ アンダーカット: {result.undercut_result.undercut_count} 箇所").style("color: #f59e0b; font-size: 0.8rem;")
        except Exception as e:
            container.clear()
            with container:
                ui.label(f"✕ エラー: {e}").style("color: #ef4444;")

    async def run_draft_analysis(container):
        """ドラフト角解析."""
        if not uploaded_files:
            ui.notify("⚠ 先にメッシュファイルをアップロードしてください", type="warning")
            return
        mesh_path = Path(uploaded_files[0]["path"])
        container.clear()
        with container:
            ui.label("⏳ ドラフト角解析中...").style("color: #94a3b8;")
        try:
            from app.mold.draft_analysis import analyze_draft_angles
            result = analyze_draft_angles(mesh_path)
            container.clear()
            with container:
                color = "#10b981" if result.compliance_ratio > 0.9 else "#f59e0b" if result.compliance_ratio > 0.5 else "#ef4444"
                ui.label("📏 ドラフト角解析結果").style(f"color: {color}; font-weight: 600;")
                ui.label(f"合格率: {result.compliance_ratio:.0%}").style(f"color: {color}; font-size: 1.1rem; font-weight: 600;")
                ui.label(f"評価面数: {len(result.faces)}, 不足: {len(result.insufficient_faces)}面").style("color: #94a3b8; font-size: 0.8rem;")
                ui.label(f"平均: {result.average_draft_deg:.1f}°, 最小: {result.min_draft_deg:.1f}°, 最大: {result.max_draft_deg:.1f}°").style("color: #94a3b8; font-size: 0.8rem;")
                if result.insufficient_faces:
                    ui.label(f"⚠ 不足面積: {result.total_insufficient_area:.1f} mm²").style("color: #f59e0b; font-size: 0.8rem;")
                    ui.label("離型時に製品の傷・固着のリスクがあります").style("color: #f59e0b; font-size: 0.75rem;")
                else:
                    ui.label("✓ 全面で十分なドラフト角が確保されています").style("color: #10b981; font-size: 0.8rem;")
        except Exception as e:
            container.clear()
            with container:
                ui.label(f"✕ エラー: {e}").style("color: #ef4444;")

    async def run_similarity(container):
        if len(uploaded_files) < 2:
            ui.notify("⚠ 類似度比較には2つのファイルが必要です", type="warning")
            return
        path_a, path_b = Path(uploaded_files[0]["path"]), Path(uploaded_files[1]["path"])
        container.clear()
        with container:
            ui.label("⏳ 類似度計算中...").style("color: #94a3b8;")
        try:
            mesh_exts = {".stl", ".obj", ".glb", ".ply"}
            if path_a.suffix.lower() in mesh_exts:
                from app.similarity.mesh_similarity import compare_mesh_files
                result = compare_mesh_files(path_a, path_b)
            else:
                from app.similarity.image_similarity import compare_image_files
                result = compare_image_files(path_a, path_b)
            container.clear()
            with container:
                ui.label(f"類似度スコア: {result.score:.1%}").style("color: #a5b4fc; font-weight: 600; font-size: 1.1rem;")
        except Exception as e:
            container.clear()
            with container:
                ui.label(f"✕ エラー: {e}").style("color: #ef4444;")

    def show_parts_list(category: str, container):
        from app.mold.parts_db import parts_db
        from dataclasses import asdict
        parts = parts_db.search(category=category)
        container.clear()
        with container:
            ui.label(f"📦 {category} — {len(parts)}件").style("color: #f1f5f9; font-weight: 600; font-size: 0.85rem;")
            for p in parts:
                with ui.card().style("background: #1e293b; border: 1px solid rgba(148,163,184,0.15); padding: 6px; border-radius: 8px;"):
                    ui.label(f"{p.name} ({p.manufacturer} {p.model_number})").style("color: #f1f5f9; font-size: 0.8rem;")
                    details = []
                    if p.diameter:
                        details.append(f"φ{p.diameter:.0f}")
                    if p.length:
                        details.append(f"L{p.length:.0f}")
                    if p.material:
                        details.append(p.material)
                    if p.unit_price > 0:
                        details.append(f"¥{p.unit_price:,.0f}")
                    ui.label(" / ".join(details)).style("color: #94a3b8; font-size: 0.7rem;")

    async def run_sizing(container):
        """サイジング計算を実行."""
        from app.mold.sizing import (
            calculate_clamp_force, design_runner, design_cooling,
            design_ejector, select_mold_steel, estimate_cycle_time,
        )
        resin = sizing_resin.value or "ABS"
        area = float(sizing_area.value or 5000)
        thickness = float(sizing_thickness.value or 2.5)
        weight = float(sizing_weight.value or 30)
        qty = int(sizing_qty.value or 10000)
        cavity = int(mold_cavity_count.value) if mold_cavity_count.value else 1

        container.clear()
        with container:
            ui.label("⏳ サイジング計算中...").style("color: #94a3b8;")
        try:
            clamp = calculate_clamp_force(area, resin, cavity)
            runner = design_runner(weight, thickness, resin, cavity)
            cooling = design_cooling(thickness, 100, 80, resin)
            ejector = design_ejector(30, area, thickness)
            steel = select_mold_steel(resin, qty)
            cycle = estimate_cycle_time(thickness, weight, resin, qty, cavity)

            container.clear()
            with container:
                ui.label("✅ サイジング結果").style("color: #10b981; font-weight: 600;")

                # 型締力
                with ui.card().style("background: #1e293b; border: 1px solid rgba(148,163,184,0.15); padding: 8px; border-radius: 8px;"):
                    ui.label("🔧 型締力").style("font-weight: 600; color: #f1f5f9; font-size: 0.8rem;")
                    ui.label(f"推奨成形機: {clamp.recommended_machine}").style("color: #a5b4fc; font-size: 0.85rem;")
                    ui.label(f"型締力: {clamp.clamp_force_ton:.1f} tf ({clamp.clamp_force_kn:.0f} kN)").style("color: #94a3b8; font-size: 0.75rem;")

                # ランナー
                with ui.card().style("background: #1e293b; border: 1px solid rgba(148,163,184,0.15); padding: 8px; border-radius: 8px;"):
                    ui.label("🔀 ランナー・ゲート").style("font-weight: 600; color: #f1f5f9; font-size: 0.8rem;")
                    ui.label(f"ゲート: {runner.gate_type} ({runner.gate_width:.1f}×{runner.gate_depth:.1f}mm)").style("color: #94a3b8; font-size: 0.75rem;")
                    ui.label(f"ランナー径: φ{runner.main_runner_diameter:.1f}, スプルー径: φ{runner.sprue_diameter:.1f}").style("color: #94a3b8; font-size: 0.75rem;")
                    ui.label(f"ランナー廃材: {runner.estimated_waste_g:.1f}g").style("color: #64748b; font-size: 0.7rem;")

                # 冷却
                with ui.card().style("background: #1e293b; border: 1px solid rgba(148,163,184,0.15); padding: 8px; border-radius: 8px;"):
                    ui.label("❄ 冷却系").style("font-weight: 600; color: #f1f5f9; font-size: 0.8rem;")
                    ui.label(f"水管: φ{cooling.channel_diameter:.0f}mm, ピッチ{cooling.channel_pitch:.0f}mm, {cooling.channel_count}本").style("color: #94a3b8; font-size: 0.75rem;")
                    ui.label(f"冷却時間: {cooling.estimated_cooling_time_s:.1f}秒, 金型温度: {cooling.mold_temperature:.0f}°C").style("color: #94a3b8; font-size: 0.75rem;")

                # 鋼材
                with ui.card().style("background: #1e293b; border: 1px solid rgba(148,163,184,0.15); padding: 8px; border-radius: 8px;"):
                    ui.label("🔩 鋼材選定").style("font-weight: 600; color: #f1f5f9; font-size: 0.8rem;")
                    ui.label(f"コア/キャビティ: {steel.core_steel}").style("color: #94a3b8; font-size: 0.75rem;")
                    ui.label(f"硬度: {steel.hardness_hrc}, 表面: {steel.surface_treatment}").style("color: #94a3b8; font-size: 0.75rem;")
                    ui.label(f"理由: {steel.reason}").style("color: #64748b; font-size: 0.7rem;")

                # サイクルタイム
                with ui.card().style("background: #1e293b; border: 1px solid rgba(148,163,184,0.15); padding: 8px; border-radius: 8px;"):
                    ui.label("⏱ サイクルタイム").style("font-weight: 600; color: #f1f5f9; font-size: 0.8rem;")
                    ui.label(f"合計: {cycle.total_cycle_s:.1f}秒 ({cycle.shots_per_hour}ショット/h)").style("color: #a5b4fc; font-size: 0.85rem;")
                    ui.label(f"射出{cycle.injection_time_s:.1f}s + 保持{cycle.hold_time_s:.1f}s + 冷却{cycle.cooling_time_s:.1f}s + 型開閉{cycle.mold_open_time_s:.1f}s").style("color: #94a3b8; font-size: 0.7rem;")
                    ui.label(f"生産所要: {cycle.production_time_hours:.1f}時間 ({qty:,}個)").style("color: #64748b; font-size: 0.7rem;")
        except Exception as e:
            container.clear()
            with container:
                ui.label(f"✕ エラー: {e}").style("color: #ef4444;")

    # ── 比較生成イベントハンドラ ────────────────

    async def run_comparison_generation(
        selection: dict[str, ui.checkbox],
        container,
    ):
        """複数エンジンで同時に3D生成し結果を比較."""
        # 選択されたエンジンを取得
        selected = [name for name, cb in selection.items() if cb.value]
        if not selected:
            ui.notify("⚠ 比較するエンジンを1つ以上選択してください", type="warning")
            return

        if not uploaded_files:
            ui.notify("⚠ 先に画像/メッシュをアップロードしてください", type="warning")
            return

        container.clear()
        with container:
            ui.label(f"⏳ {len(selected)} エンジンで並列生成中...").style(
                "color: #a5b4fc; font-weight: 600;"
            )
            for eng_name in selected:
                ui.label(f"  🔄 {eng_name}").style("color: #94a3b8; font-size: 0.8rem;")

        results: list[dict] = []

        for eng_name in selected:
            try:
                engine = EngineRegistry.get(eng_name)
                images = [
                    ProcessedImage(
                        path=Path(f["path"]),
                        view_angle=f["view_angle"],
                    )
                    for f in uploaded_files
                ]

                output_dir = settings.temp_dir / f"compare_{eng_name}_{uuid.uuid4().hex[:8]}"
                output_dir.mkdir(parents=True, exist_ok=True)

                params = ReconstructionParams(
                    output_format=OutputFormat(output_format.value),
                    output_dir=output_dir,
                )

                result = await engine.reconstruct(images, params)
                results.append({
                    "engine": eng_name,
                    "display_name": engine.get_display_name(),
                    "success": result.success,
                    "elapsed": result.elapsed_seconds,
                    "output_path": str(result.output_path) if result.output_path else None,
                    "error": result.error_message,
                    "metadata": result.metadata or {},
                })
            except Exception as e:
                results.append({
                    "engine": eng_name,
                    "display_name": eng_name,
                    "success": False,
                    "elapsed": 0.0,
                    "output_path": None,
                    "error": str(e),
                    "metadata": {},
                })

        # 結果表示
        container.clear()
        with container:
            ui.label(f"✅ 比較完了 ({len(results)} エンジン)").style(
                "color: #10b981; font-weight: 600;"
            )

            # 結果テーブル
            for r in sorted(results, key=lambda x: x.get("elapsed", 999)):
                status_icon = "✓" if r["success"] else "✕"
                status_color = "#10b981" if r["success"] else "#ef4444"
                with ui.card().style(
                    "background: #1e293b; border: 1px solid rgba(148,163,184,0.15); "
                    "padding: 10px; border-radius: 8px;"
                ):
                    with ui.row().classes("items-center justify-between w-full"):
                        ui.label(f"{status_icon} {r['display_name']}").style(
                            f"color: {status_color}; font-weight: 600; font-size: 0.85rem;"
                        )
                        ui.label(f"{r['elapsed']:.1f}秒").style(
                            "color: #a5b4fc; font-size: 0.8rem;"
                        )
                    if r["success"]:
                        meta_items = []
                        for k, v in r["metadata"].items():
                            if k != "engine":
                                meta_items.append(f"{k}: {v}")
                        if meta_items:
                            ui.label(" | ".join(meta_items[:3])).style(
                                "color: #64748b; font-size: 0.7rem;"
                            )
                        if r["output_path"]:
                            ui.button(
                                "📥 DL",
                                on_click=lambda p=r["output_path"]: ui.download(p),
                            ).props("flat dense color=indigo-4 size=sm")
                    else:
                        ui.label(f"エラー: {r['error']}").style(
                            "color: #ef4444; font-size: 0.75rem;"
                        )

    def save_settings(http_proxy: str, https_proxy: str, token: str, dialog):
        settings.http_proxy = http_proxy or None
        settings.https_proxy = https_proxy or None
        if token:
            settings.hf_token = token
        ui.notify("✓ 設定を保存しました", type="positive")
        dialog.close()


# ── エントリポイント ──────────────────────────────────

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="CAD3D Generator",
        host="127.0.0.1",
        port=8080,
        dark=True,
        reload=False,
        show=True,
        favicon="◆",
    )
