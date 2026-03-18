"""
金型構造設計モジュール.

3Dメッシュから金型の全体構造を設計する.
可動型板 (core plate) と固定型板 (cavity plate) のくり抜き、
およびモールドベース全体のレイアウトを生成.

Implements: F-072 (金型構造設計)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core.exceptions import CAD3DError
from app.mold.undercut import detect_undercuts, UndercutAnalysisResult
from app.mold.parting_line import optimize_parting_line, PartingLineResult
from app.mold.draft_analysis import analyze_draft_angles, DraftAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class MoldPlate:
    """金型プレートの情報."""
    name: str
    width: float    # mm
    height: float   # mm
    depth: float    # mm (厚さ)
    material: str = "S50C"
    weight_kg: float = 0.0


@dataclass
class MoldComponent:
    """金型構成部品."""
    name: str               # 部品名
    quantity: int = 1
    size: str = ""          # サイズ表記
    material: str = ""
    catalog_id: str = ""    # カタログ品番
    estimated_cost: float = 0.0


@dataclass
class MoldDesignResult:
    """金型設計の結果."""
    # プレート構成
    fixed_clamping_plate: MoldPlate | None = None   # 固定側取付板
    cavity_plate: MoldPlate | None = None            # 固定型板 (cavity)
    core_plate: MoldPlate | None = None              # 可動型板 (core)
    support_plate: MoldPlate | None = None           # サポートプレート
    spacer_blocks: list[MoldPlate] = field(default_factory=list)  # スペーサブロック
    ejector_plate_upper: MoldPlate | None = None     # エジェクタプレート(上)
    ejector_plate_lower: MoldPlate | None = None     # エジェクタプレート(下)
    movable_clamping_plate: MoldPlate | None = None  # 可動側取付板

    # 構成部品
    components: list[MoldComponent] = field(default_factory=list)

    # 解析結果
    undercut_result: UndercutAnalysisResult | None = None
    parting_line_result: PartingLineResult | None = None
    draft_result: DraftAnalysisResult | None = None

    # サイジング結果
    clamp_force_ton: float = 0.0
    recommended_machine: str = ""
    cycle_time_s: float = 0.0
    shots_per_hour: int = 0
    cooling_time_s: float = 0.0
    runner_waste_g: float = 0.0
    steel_grade: str = ""

    # 金型全体サイズ
    total_width: float = 0.0
    total_height: float = 0.0
    total_depth: float = 0.0
    total_weight_kg: float = 0.0

    # コスト
    estimated_material_cost: float = 0.0
    estimated_machining_cost: float = 0.0
    estimated_total_cost: float = 0.0

    summary: str = ""


class MoldDesigner:
    """金型構造設計クラス."""

    # 鋼材密度 [kg/mm³]
    STEEL_DENSITY = 7.85e-6  # S50C: 7.85 g/cm³

    # 標準マージン [mm]
    WALL_MARGIN = 40.0     # 製品外周から型板端面までの最低マージン
    DEPTH_MARGIN = 20.0    # 製品深さに対する型板厚さマージン
    PLATE_MARGIN = 10.0    # プレート間のマージン

    def design(
        self,
        mesh_path: Path,
        parting_direction: np.ndarray | None = None,
        cavity_count: int = 1,
    ) -> MoldDesignResult:
        """
        3Dメッシュから金型全体構造を設計.

        Args:
            mesh_path: 製品3Dメッシュ.
            parting_direction: 型開き方向 (None=自動).
            cavity_count: キャビティ数 (1=シングル, 2=2個取り, etc.)
        """
        try:
            import trimesh
        except ImportError:
            raise CAD3DError("trimesh が必要です")

        mesh = trimesh.load(str(mesh_path))
        if isinstance(mesh, trimesh.Scene):
            geometries = list(mesh.geometry.values())
            mesh = max(geometries, key=lambda g: len(g.faces) if hasattr(g, 'faces') else 0)

        result = MoldDesignResult()

        # Step 1: パーティングライン最適化
        logger.info("Step 1: Parting line optimization")
        pl_result = optimize_parting_line(mesh)
        result.parting_line_result = pl_result
        parting_dir = pl_result.best.direction if parting_direction is None else np.array(parting_direction)

        # Step 2: アンダーカット解析
        logger.info("Step 2: Undercut analysis")
        uc_result = detect_undercuts(mesh_path, parting_dir)
        result.undercut_result = uc_result

        # Step 3: ドラフト角解析
        logger.info("Step 3: Draft angle analysis")
        try:
            draft_result = analyze_draft_angles(mesh_path, parting_dir)
            result.draft_result = draft_result
        except Exception:
            logger.warning("Draft analysis failed", exc_info=True)

        # Step 4: 製品寸法の取得
        bb = mesh.bounding_box
        product_w, product_h, product_d = bb.extents
        product_volume = float(mesh.volume) if mesh.is_watertight else float(np.prod(bb.extents) * 0.5)

        # Step 5: プレート設計
        logger.info("Step 5: Plate sizing")
        self._design_plates(result, product_w, product_h, product_d, cavity_count)

        # Step 6: 標準部品の選定
        logger.info("Step 6: Component selection")
        self._select_components(result, product_w, product_h, product_d)

        # Step 7: サイジング統合
        logger.info("Step 7: Sizing integration")
        self._integrate_sizing(result, product_w, product_h, product_d, cavity_count)

        # Step 8: コスト概算
        logger.info("Step 8: Cost estimation")
        self._estimate_costs(result)

        # サマリ生成
        result.summary = self._generate_summary(result, product_w, product_h, product_d)
        logger.info("Mold design completed: %s", result.summary[:200])
        return result

    def _design_plates(
        self,
        result: MoldDesignResult,
        pw: float, ph: float, pd: float,
        cavity_count: int,
    ) -> None:
        """各プレートのサイズを決定."""
        # キャビティ配置を考慮した型板幅・高さ
        if cavity_count <= 1:
            plate_w = pw + 2 * self.WALL_MARGIN
            plate_h = ph + 2 * self.WALL_MARGIN
        elif cavity_count == 2:
            plate_w = 2 * pw + 3 * self.WALL_MARGIN
            plate_h = ph + 2 * self.WALL_MARGIN
        elif cavity_count == 4:
            plate_w = 2 * pw + 3 * self.WALL_MARGIN
            plate_h = 2 * ph + 3 * self.WALL_MARGIN
        else:
            cols = int(np.ceil(np.sqrt(cavity_count)))
            rows = int(np.ceil(cavity_count / cols))
            plate_w = cols * pw + (cols + 1) * self.WALL_MARGIN
            plate_h = rows * ph + (rows + 1) * self.WALL_MARGIN

        # 標準サイズに丸め (10mm単位)
        plate_w = np.ceil(plate_w / 10) * 10
        plate_h = np.ceil(plate_h / 10) * 10

        # 深さ (型開き方向)
        cavity_depth = pd * 0.55 + self.DEPTH_MARGIN  # 固定側 (キャビティ側が深め)
        core_depth = pd * 0.45 + self.DEPTH_MARGIN     # 可動側

        # 固定側取付板
        result.fixed_clamping_plate = MoldPlate(
            name="固定側取付板", width=plate_w, height=plate_h,
            depth=25.0, material="S45C",
        )

        # 固定型板 (キャビティ)
        result.cavity_plate = MoldPlate(
            name="固定型板 (Cavity)", width=plate_w, height=plate_h,
            depth=cavity_depth, material="NAK80",
        )

        # 可動型板 (コア)
        result.core_plate = MoldPlate(
            name="可動型板 (Core)", width=plate_w, height=plate_h,
            depth=core_depth, material="NAK80",
        )

        # サポートプレート
        result.support_plate = MoldPlate(
            name="サポートプレート", width=plate_w, height=plate_h,
            depth=30.0, material="S50C",
        )

        # スペーサブロック
        spacer_depth = max(40.0, pd * 0.3 + 20)
        result.spacer_blocks = [
            MoldPlate(
                name=f"スペーサブロック {side}",
                width=plate_w * 0.12, height=plate_h,
                depth=spacer_depth, material="S50C",
            )
            for side in ["左", "右"]
        ]

        # エジェクタプレート
        result.ejector_plate_upper = MoldPlate(
            name="エジェクタプレート (上)", width=plate_w * 0.7, height=plate_h * 0.7,
            depth=20.0, material="S50C",
        )
        result.ejector_plate_lower = MoldPlate(
            name="エジェクタプレート (下)", width=plate_w * 0.7, height=plate_h * 0.7,
            depth=25.0, material="S50C",
        )

        # 可動側取付板
        result.movable_clamping_plate = MoldPlate(
            name="可動側取付板", width=plate_w, height=plate_h,
            depth=25.0, material="S45C",
        )

        # 全体サイズ
        result.total_width = plate_w
        result.total_height = plate_h
        total_depth = (
            result.fixed_clamping_plate.depth +
            result.cavity_plate.depth +
            result.core_plate.depth +
            result.support_plate.depth +
            spacer_depth +
            result.ejector_plate_upper.depth +
            result.ejector_plate_lower.depth +
            result.movable_clamping_plate.depth
        )
        result.total_depth = total_depth

        # 重量計算
        for plate in self._all_plates(result):
            plate.weight_kg = plate.width * plate.height * plate.depth * self.STEEL_DENSITY
            result.total_weight_kg += plate.weight_kg

    def _select_components(
        self,
        result: MoldDesignResult,
        pw: float, ph: float, pd: float,
    ) -> None:
        """標準部品の選定."""
        plate_w = result.total_width
        plate_h = result.total_height

        # ガイドピン (4本)
        guide_pin_diameter = 20.0 if plate_w < 300 else 25.0 if plate_w < 500 else 30.0
        guide_pin_length = result.cavity_plate.depth + result.core_plate.depth + 20
        guide_pin_length = _round_up_to_standard(guide_pin_length, [60, 80, 100, 120, 150, 180, 200, 250])
        result.components.append(MoldComponent(
            name="ガイドピン",
            quantity=4,
            size=f"φ{guide_pin_diameter:.0f} × L{guide_pin_length:.0f}",
            material="SUJ2 (焼入)",
        ))

        # ガイドブッシュ (4本)
        result.components.append(MoldComponent(
            name="ガイドブッシュ",
            quantity=4,
            size=f"φ{guide_pin_diameter:.0f}",
            material="SUJ2 (焼入)",
        ))

        # 突き出しピン
        ejector_count = max(4, int((pw * ph) / 2500))  # 50mm間隔程度
        ejector_diameter = 5.0 if pd < 50 else 8.0 if pd < 100 else 10.0
        ejector_length = result.core_plate.depth + result.support_plate.depth + 10
        ejector_length = _round_up_to_standard(ejector_length, [60, 80, 100, 120, 150, 180, 200, 250, 300])
        result.components.append(MoldComponent(
            name="突き出しピン",
            quantity=ejector_count,
            size=f"φ{ejector_diameter:.0f} × L{ejector_length:.0f}",
            material="SKD61",
        ))

        # リターンピン (4本)
        return_pin_diameter = 10.0
        return_pin_length = ejector_length
        result.components.append(MoldComponent(
            name="リターンピン",
            quantity=4,
            size=f"φ{return_pin_diameter:.0f} × L{return_pin_length:.0f}",
            material="SKD61",
        ))

        # スプルーブッシュ
        result.components.append(MoldComponent(
            name="スプルーブッシュ",
            quantity=1,
            size=f"φ16 × L{result.fixed_clamping_plate.depth + result.cavity_plate.depth:.0f}",
            material="SKD61",
        ))

        # ロケートリング
        result.components.append(MoldComponent(
            name="ロケートリング",
            quantity=1,
            size="φ100 or φ150 (成形機に合わせる)",
            material="S45C",
        ))

        # ストップボルト
        result.components.append(MoldComponent(
            name="ストップボルト",
            quantity=4,
            size="M10 or M12",
            material="SCM435",
        ))

        # アンダーカットがある場合はスライドコア/傾斜コアを追加
        if result.undercut_result and result.undercut_result.has_undercut:
            for i, region in enumerate(result.undercut_result.undercut_regions):
                if region.recommended_mechanism == "slide_core":
                    result.components.append(MoldComponent(
                        name=f"スライドコアユニット #{i+1}",
                        quantity=1,
                        size=f"深さ{region.depth:.1f}mm, 面積{region.area:.0f}mm²",
                        material="NAK80",
                    ))
                    # アンギュラピン
                    result.components.append(MoldComponent(
                        name=f"アンギュラピン #{i+1}",
                        quantity=1,
                        size=f"φ12 × L80 (角度15°)",
                        material="SKD61",
                    ))
                elif region.recommended_mechanism == "angled_lifter":
                    result.components.append(MoldComponent(
                        name=f"傾斜コア #{i+1}",
                        quantity=1,
                        size=f"深さ{region.depth:.1f}mm",
                        material="NAK80",
                    ))

        # ガススプリング (エジェクタ復帰用)
        if result.total_weight_kg > 100:
            result.components.append(MoldComponent(
                name="ガススプリング",
                quantity=2,
                size="推力500N〜 (金型重量に合わせる)",
                material="—",
            ))

    def _integrate_sizing(self, result: MoldDesignResult,
                          pw: float, ph: float, pd: float,
                          cavity_count: int) -> None:
        """sizing.pyの計算結果を統合."""
        try:
            from app.mold.sizing import (
                calculate_clamp_force, design_runner, design_cooling,
                estimate_cycle_time, select_mold_steel,
            )
            # 投影面積
            proj_area = pw * ph
            # 製品重量の概算 (体積 × 樹脂密度1.05)
            part_weight = pw * ph * pd * 0.5 * 1.05 / 1000  # g
            # 肉厚概算 (製品最小寸法の30%)
            wall_t = min(pw, ph, pd) * 0.3
            wall_t = max(1.0, min(wall_t, 5.0))

            clamp = calculate_clamp_force(proj_area, "ABS", cavity_count)
            result.clamp_force_ton = clamp.clamp_force_ton
            result.recommended_machine = clamp.recommended_machine

            runner = design_runner(part_weight, wall_t, "ABS", cavity_count, result.total_width)
            result.runner_waste_g = runner.estimated_waste_g

            cooling = design_cooling(wall_t, pw, ph, "ABS")
            result.cooling_time_s = cooling.estimated_cooling_time_s

            cycle = estimate_cycle_time(wall_t, part_weight, "ABS", 10000, cavity_count)
            result.cycle_time_s = cycle.total_cycle_s
            result.shots_per_hour = cycle.shots_per_hour

            steel = select_mold_steel("ABS", 10000)
            result.steel_grade = steel.core_steel
        except Exception:
            logger.warning("Sizing integration failed", exc_info=True)

    def _estimate_costs(self, result: MoldDesignResult) -> None:
        """コスト概算."""
        # 材料費 (鋼材 ¥1,000〜3,000/kg)
        material_cost_per_kg = 1500.0  # 平均
        result.estimated_material_cost = result.total_weight_kg * material_cost_per_kg

        # 加工費 (プレート + くり抜き + 仕上げ)
        # 概算: 材料費の 3〜8倍
        complexity_factor = 4.0
        if result.undercut_result and result.undercut_result.has_undercut:
            complexity_factor += result.undercut_result.undercut_count * 0.5
        result.estimated_machining_cost = result.estimated_material_cost * complexity_factor

        # 部品費
        component_cost = len(result.components) * 3000  # 平均1部品3000円
        result.estimated_total_cost = (
            result.estimated_material_cost +
            result.estimated_machining_cost +
            component_cost
        )

    def _all_plates(self, result: MoldDesignResult) -> list[MoldPlate]:
        """全プレートのリスト."""
        plates = []
        for attr in [
            result.fixed_clamping_plate, result.cavity_plate,
            result.core_plate, result.support_plate,
            result.ejector_plate_upper, result.ejector_plate_lower,
            result.movable_clamping_plate,
        ]:
            if attr:
                plates.append(attr)
        plates.extend(result.spacer_blocks)
        return plates

    def _generate_summary(
        self, result: MoldDesignResult,
        pw: float, ph: float, pd: float,
    ) -> str:
        lines = [
            f"▼ 金型設計結果",
            f"  製品サイズ: {pw:.1f} × {ph:.1f} × {pd:.1f} mm",
            f"  金型サイズ: {result.total_width:.0f} × {result.total_height:.0f} × {result.total_depth:.0f} mm",
            f"  金型重量: {result.total_weight_kg:.1f} kg",
            f"  構成部品数: {len(result.components)}",
        ]
        if result.undercut_result and result.undercut_result.has_undercut:
            lines.append(f"  アンダーカット: {result.undercut_result.undercut_count} 箇所")
        else:
            lines.append(f"  アンダーカット: なし")
        if result.recommended_machine:
            lines.append(f"  推奨成形機: {result.recommended_machine}")
        if result.cycle_time_s > 0:
            lines.append(f"  サイクルタイム: {result.cycle_time_s:.1f}秒 ({result.shots_per_hour}ショット/h)")
        if result.draft_result:
            lines.append(f"  ドラフト角合格率: {result.draft_result.compliance_ratio:.0%}")
        lines.append(f"  概算コスト: ¥{result.estimated_total_cost:,.0f}")
        return "\n".join(lines)


def _round_up_to_standard(value: float, standards: list[float]) -> float:
    """値を標準サイズに切り上げ."""
    for s in sorted(standards):
        if s >= value:
            return s
    return standards[-1]
