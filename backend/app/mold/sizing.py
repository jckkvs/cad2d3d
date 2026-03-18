"""
ルールベースサイジングモジュール.

射出成形金型の各パラメータをルールベースで計算する.
見積に影響する要素に焦点を当て、
機械加工費・素材費・標準部品費を概算するためのロジック.

計算対象:
1. 射出成形機のトン数 (型締力)
2. ランナー・ゲートの寸法
3. 冷却水管の配置ルール
4. エジェクタストローク・突き出し力
5. 金型鋼材の選定ルール

参考文献:
- "Injection Mold Design Engineering" (David O. Kazmer)
  原文: "The required clamp force is dependent on the projected area
  of the part and the mold cavity pressure."
  訳: 必要な型締力は、製品の投影面積とキャビティ内圧力に依存する.

- "Mold Engineering" (Herbert Rees)
  原文: "The runner system ... must be designed to deliver molten
  plastic to the mold cavities at the proper temperature and pressure."
  訳: ランナーシステムは適切な温度と圧力でキャビティに溶融樹脂を送達する設計が必要.

Implements: F-074 (ルールベースサイジング)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ===== 樹脂材料データベース =====

class ResinType(str, Enum):
    """代表的な樹脂材料."""
    PP = "PP"         # ポリプロピレン
    PE = "PE"         # ポリエチレン
    ABS = "ABS"       # ABS
    PA = "PA"         # ナイロン (PA6, PA66)
    POM = "POM"       # ポリアセタール
    PC = "PC"         # ポリカーボネート
    PBT = "PBT"       # PBT
    PS = "PS"         # ポリスチレン
    PMMA = "PMMA"     # アクリル
    PPS = "PPS"       # PPS


@dataclass
class ResinProperties:
    """樹脂の物性."""
    name: str
    density: float            # g/cm³
    melt_temp_min: float      # 溶融温度下限 [°C]
    melt_temp_max: float      # 溶融温度上限 [°C]
    mold_temp_min: float      # 金型温度下限 [°C]
    mold_temp_max: float      # 金型温度上限 [°C]
    injection_pressure: float # 射出圧力 [MPa]
    shrinkage_min: float      # 成形収縮率下限 [%]
    shrinkage_max: float      # 成形収縮率上限 [%]
    flow_ratio: float         # L/t比 (流動長/肉厚)
    unit_price_per_kg: float  # 原料単価 [円/kg]


# 樹脂物性データベース
RESIN_DB: dict[str, ResinProperties] = {
    "PP": ResinProperties("ポリプロピレン", 0.91, 200, 280, 20, 60, 80, 1.0, 2.5, 200, 150),
    "PE": ResinProperties("ポリエチレン", 0.95, 180, 260, 20, 50, 70, 1.5, 3.5, 180, 120),
    "ABS": ResinProperties("ABS", 1.05, 210, 270, 50, 80, 100, 0.4, 0.7, 150, 250),
    "PA": ResinProperties("ナイロン", 1.14, 250, 300, 70, 100, 120, 0.7, 2.0, 120, 500),
    "POM": ResinProperties("ポリアセタール", 1.41, 180, 210, 60, 100, 100, 1.8, 2.5, 100, 400),
    "PC": ResinProperties("ポリカーボネート", 1.20, 280, 320, 80, 120, 130, 0.5, 0.7, 100, 600),
    "PBT": ResinProperties("PBT", 1.31, 230, 270, 60, 80, 100, 1.3, 2.0, 120, 450),
    "PS": ResinProperties("ポリスチレン", 1.05, 180, 260, 30, 60, 80, 0.3, 0.6, 200, 180),
    "PMMA": ResinProperties("アクリル", 1.19, 220, 280, 50, 80, 120, 0.3, 0.6, 130, 350),
    "PPS": ResinProperties("PPS", 1.35, 300, 340, 120, 150, 140, 0.2, 1.0, 80, 2000),
}


# ===== 型締力計算 =====

@dataclass
class ClampForceResult:
    """型締力計算の結果."""
    projected_area_cm2: float   # 投影面積 [cm²]
    cavity_pressure_mpa: float  # キャビティ内圧 [MPa]
    clamp_force_kn: float       # 型締力 [kN]
    clamp_force_ton: float      # 型締力 [tf]
    recommended_machine: str    # 推奨成形機トン数
    safety_factor: float


def calculate_clamp_force(
    projected_area_mm2: float,
    resin: str = "ABS",
    cavity_count: int = 1,
    safety_factor: float = 1.2,
) -> ClampForceResult:
    """
    必要型締力を計算.

    F_clamp = A_proj × P_cavity × n_cavity × S_factor

    Kazmer (2007):
    "The required clamp force is dependent on the projected area
    of the part and the mold cavity pressure."
    訳: 必要型締力 = 投影面積 × キャビティ内圧.

    Args:
        projected_area_mm2: 型開き方向の投影面積 [mm²].
        resin: 樹脂種類.
        cavity_count: キャビティ数.
        safety_factor: 安全率 (1.1〜1.3).

    Returns:
        型締力計算結果.
    """
    props = RESIN_DB.get(resin)
    if props is None:
        props = RESIN_DB["ABS"]
        logger.warning("Unknown resin '%s', using ABS defaults", resin)

    area_cm2 = projected_area_mm2 / 100.0  # mm² → cm²
    pressure = props.injection_pressure  # MPa

    # 型締力 [kN] = 投影面積[cm²] × 圧力[MPa] × (MPa→kgf/cm²: ×10.197) / 1000
    # 簡略: F[kN] = A[cm²] × P[MPa] × 0.1
    clamp_kn = area_cm2 * pressure * 0.1 * cavity_count * safety_factor
    clamp_ton = clamp_kn / 9.80665

    # 推奨成形機
    machine = _recommend_machine_tonnage(clamp_ton)

    return ClampForceResult(
        projected_area_cm2=area_cm2,
        cavity_pressure_mpa=pressure,
        clamp_force_kn=clamp_kn,
        clamp_force_ton=clamp_ton,
        recommended_machine=machine,
        safety_factor=safety_factor,
    )


def _recommend_machine_tonnage(required_ton: float) -> str:
    """標準成形機トン数から推奨を選択."""
    standard = [50, 80, 100, 130, 150, 180, 200, 250, 300, 350, 450,
                550, 650, 850, 1000, 1300, 1600, 2000, 2500, 3500]
    for t in standard:
        if t >= required_ton:
            return f"{t}t"
    return f"{standard[-1]}t以上"


# ===== ランナー設計 =====

@dataclass
class RunnerDesignResult:
    """ランナー設計結果."""
    sprue_diameter: float         # スプルー径 [mm]
    main_runner_diameter: float   # メインランナー径 [mm]
    sub_runner_diameter: float    # サブランナー径 [mm]
    gate_type: str                # ゲート種類
    gate_width: float             # ゲート幅 [mm]
    gate_depth: float             # ゲート深さ [mm]
    runner_length_total: float    # ランナー長さ合計 [mm]
    estimated_waste_g: float      # ランナー廃材量 [g]


def design_runner(
    part_weight_g: float,
    wall_thickness_mm: float,
    resin: str = "ABS",
    cavity_count: int = 1,
    mold_width_mm: float = 300,
) -> RunnerDesignResult:
    """
    ランナー・ゲートを設計.

    Rees (2002):
    "The runner system ... must be designed to deliver molten plastic
    to the mold cavities at the proper temperature and pressure."
    訳: ランナーは適切な温度・圧力で溶融樹脂を送達する設計が必要.

    ルール:
    - スプルー径: ゲート径 + 1〜2mm, 最小3mm
    - メインランナー径: √(製品重量 × 4 / π / L)^4 (概算)
    - ゲート幅: 肉厚 × 0.5〜1.0
    """
    props = RESIN_DB.get(resin, RESIN_DB["ABS"])

    # ゲート寸法
    gate_depth = wall_thickness_mm * 0.6
    gate_width = wall_thickness_mm * 0.8
    gate_depth = max(0.5, min(gate_depth, 3.0))
    gate_width = max(0.8, min(gate_width, 5.0))

    # ゲート種類の選定
    if wall_thickness_mm < 1.5:
        gate_type = "ピンゲート"
    elif wall_thickness_mm < 3.0:
        gate_type = "サイドゲート"
    else:
        gate_type = "ダイレクトゲート" if cavity_count == 1 else "サイドゲート"

    # ランナー径
    # 経験則: D_runner ≈ 肉厚 × 1.5 + 2mm, 最小4mm
    main_runner_d = max(4.0, wall_thickness_mm * 1.5 + 2.0)
    sub_runner_d = main_runner_d * 0.8

    # スプルー径
    sprue_d = max(3.0, main_runner_d + 1.0)

    # ランナー長さ概算
    if cavity_count <= 1:
        runner_length = mold_width_mm * 0.3
    elif cavity_count == 2:
        runner_length = mold_width_mm * 0.5
    else:
        runner_length = mold_width_mm * 0.7

    # ランナー廃材量
    runner_volume = math.pi * (main_runner_d / 2) ** 2 * runner_length  # mm³
    sprue_volume = math.pi * (sprue_d / 2) ** 2 * 60  # スプルー長60mm仮定
    waste_volume = runner_volume + sprue_volume  # mm³
    waste_g = waste_volume * props.density / 1000  # g

    return RunnerDesignResult(
        sprue_diameter=sprue_d,
        main_runner_diameter=main_runner_d,
        sub_runner_diameter=sub_runner_d,
        gate_type=gate_type,
        gate_width=gate_width,
        gate_depth=gate_depth,
        runner_length_total=runner_length,
        estimated_waste_g=waste_g,
    )


# ===== 冷却系設計 =====

@dataclass
class CoolingDesignResult:
    """冷却系設計結果."""
    channel_diameter: float     # 冷却水管径 [mm]
    channel_pitch: float        # 冷却水管ピッチ [mm]
    channel_depth: float        # 表面からの深さ [mm]
    channel_count: int          # 水管数 (1面あたり)
    estimated_cooling_time_s: float  # 冷却時間 [秒]
    mold_temperature: float     # 推奨金型温度 [°C]


def design_cooling(
    wall_thickness_mm: float,
    part_width_mm: float,
    part_length_mm: float,
    resin: str = "ABS",
) -> CoolingDesignResult:
    """
    冷却水管を設計.

    ルール (実務経験則):
    - 水管径: 8〜12mm (標準10mm)
    - 水管ピッチ: 水管径の 3〜5倍
    - 表面距離: 水管径の 1.5〜2倍
    - 冷却時間: t_cool ≈ (s²/π²α) × ln(8T_melt/(π²T_eject))
      s = 肉厚/2, α = 熱拡散率
    """
    props = RESIN_DB.get(resin, RESIN_DB["ABS"])

    # 水管径
    if wall_thickness_mm < 2.0:
        ch_diameter = 8.0
    elif wall_thickness_mm < 4.0:
        ch_diameter = 10.0
    else:
        ch_diameter = 12.0

    # ピッチと深さ
    ch_pitch = ch_diameter * 3.5
    ch_depth = ch_diameter * 1.8

    # 水管数
    count_w = max(2, int(part_width_mm / ch_pitch))
    count_l = max(2, int(part_length_mm / ch_pitch))
    ch_count = count_w + count_l

    # 冷却時間の概算 (一次元非定常伝熱: Fourier)
    # α ≈ 1.0e-7 m²/s (樹脂の典型値)
    alpha = 1.0e-7  # m²/s
    s = wall_thickness_mm / 2 / 1000  # m
    t_melt = (props.melt_temp_min + props.melt_temp_max) / 2
    t_eject = props.mold_temp_max + 10  # 離型温度

    if t_eject > 0 and t_melt > t_eject:
        cooling_time = (s ** 2 / (math.pi ** 2 * alpha)) * math.log(
            8 * (t_melt - props.mold_temp_min) / (math.pi ** 2 * (t_eject - props.mold_temp_min))
        )
    else:
        cooling_time = 15.0  # フォールバック

    cooling_time = max(3.0, min(cooling_time, 120.0))
    mold_temp = (props.mold_temp_min + props.mold_temp_max) / 2

    return CoolingDesignResult(
        channel_diameter=ch_diameter,
        channel_pitch=ch_pitch,
        channel_depth=ch_depth,
        channel_count=ch_count,
        estimated_cooling_time_s=cooling_time,
        mold_temperature=mold_temp,
    )


# ===== エジェクタ設計 =====

@dataclass
class EjectorDesignResult:
    """エジェクタ設計結果."""
    ejector_stroke_mm: float      # エジェクタストローク [mm]
    ejector_force_kn: float       # 突き出し力 [kN]
    ejector_pin_count: int        # 突き出しピン数
    ejector_pin_diameter: float   # ピン径 [mm]
    needs_stripper_plate: bool    # ストリッパプレート必要性
    needs_air_valve: bool         # エアバルブ必要性


def design_ejector(
    part_depth_mm: float,
    part_area_mm2: float,
    wall_thickness_mm: float,
    draft_angle_deg: float = 1.0,
    resin: str = "ABS",
) -> EjectorDesignResult:
    """
    エジェクタ系を設計.

    ルール:
    - ストローク ≥ 製品深さ + 10mm
    - 突き出し力: F = μ × P_res × A_side
      μ = 摩擦係数, P_res = 残留圧力, A_side = 側面積
    - ピン数 ≈ 投影面積 / 2500mm² (50mm間隔)
    - 抜き勾配 < 0.5° → ストリッパプレート推奨
    """
    # ストローク
    stroke = part_depth_mm + 10.0
    stroke = max(20.0, stroke)

    # 突き出し力 (概算)
    friction = 0.3  # 樹脂-鋼の摩擦係数
    residual_pressure = 5.0  # MPa (残留圧力)
    # 側面積 (概算: 周囲長 × 深さ)
    perimeter = math.sqrt(part_area_mm2) * 4  # 正方形近似
    side_area = perimeter * part_depth_mm
    eject_force = friction * residual_pressure * side_area / 1000  # kN

    # ピン数
    pin_count = max(4, int(part_area_mm2 / 2500))

    # ピン径
    if wall_thickness_mm < 1.5:
        pin_d = 3.0
    elif wall_thickness_mm < 3.0:
        pin_d = 5.0
    else:
        pin_d = 8.0

    # ストリッパプレート
    needs_stripper = draft_angle_deg < 0.5 or wall_thickness_mm < 1.0

    # エアバルブ (大面積 or 深い製品)
    needs_air = part_depth_mm > 30 or part_area_mm2 > 10000

    return EjectorDesignResult(
        ejector_stroke_mm=stroke,
        ejector_force_kn=eject_force,
        ejector_pin_count=pin_count,
        ejector_pin_diameter=pin_d,
        needs_stripper_plate=needs_stripper,
        needs_air_valve=needs_air,
    )


# ===== 金型鋼材選定 =====

@dataclass
class SteelSelection:
    """金型鋼材の選定結果."""
    core_steel: str           # コア鋼材
    cavity_steel: str         # キャビティ鋼材
    base_steel: str           # ベース鋼材
    hardness_hrc: str         # 必要硬度
    surface_treatment: str    # 表面処理
    reason: str


def select_mold_steel(
    resin: str = "ABS",
    production_quantity: int = 10000,
    surface_finish: str = "standard",
) -> SteelSelection:
    """
    生産量・樹脂・仕上げに基づく鋼材選定.

    ルール:
    - 少量 (< 5,000): S50C (安価)
    - 中量 (5,000〜50,000): NAK80 (プリハードン)
    - 大量 (> 50,000): SKD61 or STAVAX (焼入)
    - ガラス入り樹脂: 耐摩耗鋼 (ELMAX等)
    - 鏡面仕上げ: STAVAX
    """
    props = RESIN_DB.get(resin, RESIN_DB["ABS"])

    abrasive = resin in ("PA", "PBT", "PPS")  # ガラス繊維入りが多い
    mirror = surface_finish == "mirror"

    if mirror:
        core_steel = "STAVAX (SUS420J2相当)"
        cavity_steel = "STAVAX"
        hardness = "HRC 50-54"
        surface = "鏡面研磨 Ra0.02"
        reason = "鏡面仕上げ要求のため高耐食鋼を選定"
    elif abrasive and production_quantity > 10000:
        core_steel = "ELMAX / SKD11"
        cavity_steel = "ELMAX / SKD11"
        hardness = "HRC 58-62"
        surface = "窒化処理 or TiNコーティング"
        reason = "ガラス繊維入り樹脂の高耐摩耗性が必要"
    elif production_quantity > 50000:
        core_steel = "SKD61 (焼入)"
        cavity_steel = "SKD61 (焼入)"
        hardness = "HRC 48-52"
        surface = "窒化処理"
        reason = "大量生産のため焼入鋼で耐久性確保"
    elif production_quantity > 5000:
        core_steel = "NAK80"
        cavity_steel = "NAK80"
        hardness = "HRC 37-43"
        surface = "標準仕上げ"
        reason = "中量生産でプリハードン鋼が最適"
    else:
        core_steel = "S50C"
        cavity_steel = "S50C"
        hardness = "HRC 15-20"
        surface = "標準仕上げ"
        reason = "少量生産で安価な鋼材で十分"

    return SteelSelection(
        core_steel=core_steel,
        cavity_steel=cavity_steel,
        base_steel="S50C",
        hardness_hrc=hardness,
        surface_treatment=surface,
        reason=reason,
    )


# ===== 成形サイクルタイム =====

@dataclass
class CycleTimeResult:
    """成形サイクルタイム結果."""
    injection_time_s: float
    hold_time_s: float
    cooling_time_s: float
    mold_open_time_s: float
    total_cycle_s: float
    shots_per_hour: int
    production_time_hours: float  # 目標生産数に対する時間


def estimate_cycle_time(
    wall_thickness_mm: float,
    part_weight_g: float,
    resin: str = "ABS",
    production_quantity: int = 10000,
    cavity_count: int = 1,
) -> CycleTimeResult:
    """
    成形サイクルタイムを概算.

    ルール:
    - 射出時間 ≈ 1〜3秒 (小物) / 3〜10秒 (大物)
    - 保持時間 ≈ 射出時間 × 2〜3
    - 冷却時間: 肉厚依存 (Fourier法)
    - 型開閉 ≈ 3〜8秒
    """
    cooling = design_cooling(wall_thickness_mm, 100, 100, resin)

    # 射出時間 (重量ベース)
    injection_time = max(1.0, min(10.0, part_weight_g * 0.05))

    # 保持時間
    hold_time = injection_time * 2.5

    # 型開閉
    mold_open = 5.0

    total = injection_time + hold_time + cooling.estimated_cooling_time_s + mold_open
    shots_per_hour = int(3600 / total) if total > 0 else 0
    effective_shots = shots_per_hour * cavity_count

    production_hours = production_quantity / effective_shots if effective_shots > 0 else float("inf")

    return CycleTimeResult(
        injection_time_s=injection_time,
        hold_time_s=hold_time,
        cooling_time_s=cooling.estimated_cooling_time_s,
        mold_open_time_s=mold_open,
        total_cycle_s=total,
        shots_per_hour=shots_per_hour,
        production_time_hours=production_hours,
    )
