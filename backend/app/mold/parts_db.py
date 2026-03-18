"""
標準部品データベース.

金型に使用する標準部品 (ガイドピン、突き出しピン、ガススプリング等) の
メーカー・型式・サイズ・材質・価格を管理.

Implements: F-073 (部品DB + 自動推奨)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MoldPart:
    """金型標準部品のデータ."""
    id: str                       # 一意ID (例: "GP-MISUMI-SFJ-20-100")
    category: str                 # カテゴリ (guide_pin, ejector_pin, etc.)
    name: str                     # 部品名
    manufacturer: str             # メーカー
    model_number: str             # 型式
    diameter: float | None = None # 径 [mm]
    length: float | None = None   # 長さ [mm]
    material: str = ""
    hardness: str = ""            # 硬度 (HRC等)
    unit_price: float = 0.0       # 単価 [円]
    lead_time_days: int = 0       # 納期 [日]
    url: str = ""                 # カタログURL
    notes: str = ""


PART_CATEGORIES = {
    "guide_pin": "ガイドピン",
    "guide_bushing": "ガイドブッシュ",
    "ejector_pin": "突き出しピン",
    "return_pin": "リターンピン",
    "sprue_bushing": "スプルーブッシュ",
    "locating_ring": "ロケートリング",
    "angular_pin": "アンギュラピン",
    "gas_spring": "ガススプリング",
    "coil_spring": "コイルスプリング",
    "support_pillar": "サポートピラー",
    "parting_lock": "パーティングロック",
    "cooling_plug": "冷却プラグ",
    "o_ring": "Oリング",
    "stop_bolt": "ストップボルト",
}


# デフォルト部品カタログ (代表的なサイズ)
_DEFAULT_CATALOG: list[dict[str, Any]] = [
    # ===== ガイドピン =====
    {"id": "GP-01", "category": "guide_pin", "name": "ストレートガイドピン",
     "manufacturer": "MISUMI", "model_number": "SFJ",
     "diameter": 16, "length": 80, "material": "SUJ2", "hardness": "HRC58-62",
     "unit_price": 1200},
    {"id": "GP-02", "category": "guide_pin", "name": "ストレートガイドピン",
     "manufacturer": "MISUMI", "model_number": "SFJ",
     "diameter": 16, "length": 100, "material": "SUJ2", "hardness": "HRC58-62",
     "unit_price": 1400},
    {"id": "GP-03", "category": "guide_pin", "name": "ストレートガイドピン",
     "manufacturer": "MISUMI", "model_number": "SFJ",
     "diameter": 20, "length": 100, "material": "SUJ2", "hardness": "HRC58-62",
     "unit_price": 1800},
    {"id": "GP-04", "category": "guide_pin", "name": "ストレートガイドピン",
     "manufacturer": "MISUMI", "model_number": "SFJ",
     "diameter": 20, "length": 120, "material": "SUJ2", "hardness": "HRC58-62",
     "unit_price": 2100},
    {"id": "GP-05", "category": "guide_pin", "name": "ストレートガイドピン",
     "manufacturer": "MISUMI", "model_number": "SFJ",
     "diameter": 20, "length": 150, "material": "SUJ2", "hardness": "HRC58-62",
     "unit_price": 2500},
    {"id": "GP-06", "category": "guide_pin", "name": "ストレートガイドピン",
     "manufacturer": "MISUMI", "model_number": "SFJ",
     "diameter": 25, "length": 150, "material": "SUJ2", "hardness": "HRC58-62",
     "unit_price": 3200},
    {"id": "GP-07", "category": "guide_pin", "name": "ストレートガイドピン",
     "manufacturer": "MISUMI", "model_number": "SFJ",
     "diameter": 25, "length": 200, "material": "SUJ2", "hardness": "HRC58-62",
     "unit_price": 3800},
    {"id": "GP-08", "category": "guide_pin", "name": "ストレートガイドピン",
     "manufacturer": "MISUMI", "model_number": "SFJ",
     "diameter": 30, "length": 200, "material": "SUJ2", "hardness": "HRC58-62",
     "unit_price": 4500},

    # ===== 突き出しピン =====
    {"id": "EP-01", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 3, "length": 60, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 350},
    {"id": "EP-02", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 3, "length": 80, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 400},
    {"id": "EP-03", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 5, "length": 80, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 450},
    {"id": "EP-04", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 5, "length": 100, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 550},
    {"id": "EP-05", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 5, "length": 120, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 650},
    {"id": "EP-06", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 8, "length": 100, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 700},
    {"id": "EP-07", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 8, "length": 150, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 900},
    {"id": "EP-08", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 10, "length": 150, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 1100},
    {"id": "EP-09", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 10, "length": 200, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 1400},
    {"id": "EP-10", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 10, "length": 250, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 1700},
    {"id": "EP-11", "category": "ejector_pin", "name": "突き出しピン (ストレート)",
     "manufacturer": "MISUMI", "model_number": "EPS",
     "diameter": 10, "length": 300, "material": "SKD61", "hardness": "HRC44-48",
     "unit_price": 2000},

    # ===== ガススプリング =====
    {"id": "GS-01", "category": "gas_spring", "name": "ガススプリング",
     "manufacturer": "KALLER", "model_number": "X350",
     "diameter": 25, "length": 50, "material": "—", "unit_price": 8000,
     "notes": "推力350N"},
    {"id": "GS-02", "category": "gas_spring", "name": "ガススプリング",
     "manufacturer": "KALLER", "model_number": "X750",
     "diameter": 32, "length": 63, "material": "—", "unit_price": 12000,
     "notes": "推力750N"},
    {"id": "GS-03", "category": "gas_spring", "name": "ガススプリング",
     "manufacturer": "KALLER", "model_number": "X1500",
     "diameter": 38, "length": 80, "material": "—", "unit_price": 18000,
     "notes": "推力1500N"},
    {"id": "GS-04", "category": "gas_spring", "name": "ガススプリング",
     "manufacturer": "KALLER", "model_number": "X3000",
     "diameter": 50, "length": 100, "material": "—", "unit_price": 25000,
     "notes": "推力3000N"},

    # ===== スプルーブッシュ =====
    {"id": "SB-01", "category": "sprue_bushing", "name": "スプルーブッシュ",
     "manufacturer": "MISUMI", "model_number": "CSP",
     "diameter": 16, "length": 60, "material": "SKD61", "hardness": "HRC50-54",
     "unit_price": 2800},
    {"id": "SB-02", "category": "sprue_bushing", "name": "スプルーブッシュ",
     "manufacturer": "MISUMI", "model_number": "CSP",
     "diameter": 16, "length": 80, "material": "SKD61", "hardness": "HRC50-54",
     "unit_price": 3200},
    {"id": "SB-03", "category": "sprue_bushing", "name": "スプルーブッシュ",
     "manufacturer": "MISUMI", "model_number": "CSP",
     "diameter": 16, "length": 100, "material": "SKD61", "hardness": "HRC50-54",
     "unit_price": 3600},
]


class PartsDatabase:
    """標準部品データベース."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or (settings.data_dir / "parts_catalog.json")
        self._parts: list[MoldPart] = []
        self._load()

    def _load(self) -> None:
        """DBをロード (なければデフォルト作成)."""
        if self._db_path.exists():
            try:
                data = json.loads(self._db_path.read_text(encoding="utf-8"))
                self._parts = [MoldPart(**p) for p in data]
                return
            except Exception:
                logger.warning("DB read failed, using default")

        # デフォルトカタログで初期化
        self._parts = [MoldPart(**p) for p in _DEFAULT_CATALOG]
        self.save()

    def save(self) -> None:
        """DBを保存."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(p) for p in self._parts]
        self._db_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def search(
        self,
        category: str | None = None,
        min_diameter: float | None = None,
        min_length: float | None = None,
        manufacturer: str | None = None,
    ) -> list[MoldPart]:
        """部品を検索."""
        results = self._parts
        if category:
            results = [p for p in results if p.category == category]
        if min_diameter is not None:
            results = [p for p in results if p.diameter and p.diameter >= min_diameter]
        if min_length is not None:
            results = [p for p in results if p.length and p.length >= min_length]
        if manufacturer:
            results = [p for p in results if manufacturer.lower() in p.manufacturer.lower()]
        return results

    def recommend(
        self,
        category: str,
        required_diameter: float | None = None,
        required_length: float | None = None,
    ) -> MoldPart | None:
        """
        必要サイズに最も近い部品を推奨.

        戦略: 必要サイズ以上で最小の部品を選択.
        """
        candidates = self.search(
            category=category,
            min_diameter=required_diameter,
            min_length=required_length,
        )
        if not candidates:
            return None

        # 径→長さの優先度でソート (最小の適合品)
        candidates.sort(key=lambda p: (p.diameter or 0, p.length or 0))
        return candidates[0]

    def add_part(self, part: MoldPart) -> None:
        """部品を追加."""
        # 重複ID チェック
        self._parts = [p for p in self._parts if p.id != part.id]
        self._parts.append(part)
        self.save()

    def remove_part(self, part_id: str) -> bool:
        """部品を削除."""
        original = len(self._parts)
        self._parts = [p for p in self._parts if p.id != part_id]
        if len(self._parts) < original:
            self.save()
            return True
        return False

    def list_categories(self) -> dict[str, str]:
        """カテゴリ一覧."""
        return PART_CATEGORIES.copy()

    def list_all(self) -> list[MoldPart]:
        """全部品."""
        return list(self._parts)


# シングルトン
parts_db = PartsDatabase()
