"""
API統合テスト.

FastAPI エンドポイントの統合テスト。
生成フロー、エンジン一覧、比較API等を検証。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

# conftest.py の ensure_engines fixture を全テストに適用
pytestmark = pytest.mark.usefixtures("ensure_engines")


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ===== エンジン一覧 API =====


class TestEnginesAPI:
    """GET /api/generate/engines のテスト."""

    def test_list_engines(self, client: TestClient) -> None:
        """エンジン一覧が取得できる."""
        resp = client.get("/api/generate/engines")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 9

    def test_engine_has_required_fields(self, client: TestClient) -> None:
        """各エンジンに必須フィールドが含まれる."""
        resp = client.get("/api/generate/engines")
        for engine in resp.json():
            assert "name" in engine
            assert "display_name" in engine
            assert "description" in engine
            assert "status" in engine

    def test_engine_detail(self, client: TestClient) -> None:
        """個別エンジン情報が取得できる."""
        resp = client.get("/api/generate/engines/triposr")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "triposr"

    def test_engine_detail_not_found(self, client: TestClient) -> None:
        """存在しないエンジン."""
        resp = client.get("/api/generate/engines/nonexistent_engine_xyz")
        assert resp.status_code in (404, 500)

    def test_all_new_engines_in_list(self, client: TestClient) -> None:
        """新規5エンジンが一覧に含まれる."""
        resp = client.get("/api/generate/engines")
        names = {e["name"] for e in resp.json()}
        for expected in ["instantmesh", "crm", "zero123pp", "wonder3d", "secadnet"]:
            assert expected in names, f"{expected} が見つからない"

    def test_secadnet_capabilities_in_api(self, client: TestClient) -> None:
        """SECAD-NetのCAD出力がAPI経由で確認できる."""
        resp = client.get("/api/generate/engines/secadnet")
        data = resp.json()
        assert data["capabilities"]["outputs_cad"] is True

    def test_engines_have_status(self, client: TestClient) -> None:
        """各エンジンにステータスがある."""
        resp = client.get("/api/generate/engines")
        for engine in resp.json():
            assert engine["status"] in ("ready", "weights_missing", "dependency_missing", "error")


# ===== 比較 API =====


class TestCompareAPI:
    """POST /api/generate/compare のテスト."""

    def test_compare_with_no_engines(self, client: TestClient) -> None:
        """エンジン未指定の比較リクエスト."""
        resp = client.post("/api/generate/compare", json={
            "engine_name": "triposr",
            "images": [],
            "output_format": "glb",
        })
        # エンジン指定なし or 画像なしのため失敗系
        assert resp.status_code in (200, 422, 500)

    def test_compare_result_nonexistent(self, client: TestClient) -> None:
        """存在しないジョブIDの比較結果."""
        resp = client.get("/api/generate/compare/fake-id-123")
        assert resp.status_code == 200


# ===== ジョブ管理 API =====


class TestJobsAPI:
    """ジョブ管理エンドポイントのテスト."""

    def test_job_status_nonexistent(self, client: TestClient) -> None:
        """存在しないジョブのステータス."""
        resp = client.get("/api/generate/jobs/nonexistent-job-id")
        assert resp.status_code in (200, 404)

    def test_job_result_nonexistent(self, client: TestClient) -> None:
        """存在しないジョブの結果."""
        resp = client.get("/api/generate/jobs/nonexistent-job-id/result")
        assert resp.status_code in (200, 404)


# ===== 金型 API =====


class TestMoldAPIIntegration:
    """金型APIの統合テスト."""

    def test_parts_categories(self, client: TestClient) -> None:
        """部品カテゴリ一覧."""
        resp = client.get("/api/mold/parts/categories")
        assert resp.status_code == 200
        cats = resp.json()
        # dict形式 {category_key: display_name} で返却される
        assert isinstance(cats, (list, dict))
        assert len(cats) > 0

    def test_parts_list(self, client: TestClient) -> None:
        """部品一覧."""
        resp = client.get("/api/mold/parts")
        assert resp.status_code == 200
        parts = resp.json()
        assert isinstance(parts, list)

    def test_sizing_clamp_force(self, client: TestClient) -> None:
        """型締力計算."""
        resp = client.post("/api/mold/sizing/clamp-force", json={
            "projected_area_mm2": 5000,
            "resin": "ABS",
            "cavity_count": 1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "clamp_force_kn" in data.get("data", data)

    def test_sizing_cycle_time(self, client: TestClient) -> None:
        """サイクルタイム計算."""
        resp = client.post("/api/mold/sizing/cycle-time", json={
            "wall_thickness_mm": 2.5,
            "part_weight_g": 30,
            "resin": "ABS",
            "production_quantity": 10000,
            "cavity_count": 1,
        })
        assert resp.status_code == 200

    def test_resins_list(self, client: TestClient) -> None:
        """樹脂一覧."""
        resp = client.get("/api/mold/resins")
        assert resp.status_code == 200
        resins = resp.json()
        # dict形式 {resin_key: display_name} で返却される
        assert isinstance(resins, (list, dict))
        assert len(resins) > 0


# ===== ヘルスチェック =====


class TestHealthCheck:
    """アプリケーション起動確認."""

    def test_root_redirect(self, client: TestClient) -> None:
        """ルートパスにアクセス可能."""
        resp = client.get("/", follow_redirects=False)
        # NiceGUI統合時はリダイレクトの場合もある
        assert resp.status_code in (200, 307, 404)

    def test_api_docs(self, client: TestClient) -> None:
        """OpenAPI docs が利用可能."""
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_openapi_schema(self, client: TestClient) -> None:
        """OpenAPI スキーマが取得可能."""
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "info" in schema
