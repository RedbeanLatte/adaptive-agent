from __future__ import annotations

import json
from pathlib import Path


def _fixture_path(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / name


def test_roblox_coin_pickup_request_fixture_exists():
    path = _fixture_path("roblox_coin_pickup_request.txt")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "coin" in text.lower()
    assert "leaderstats" in text
    assert "Touched" in text


def test_material_lava_cave_brief_fixture_exists():
    path = _fixture_path("material_lava_cave_brief.txt")
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "stylized lava cave" in text.lower()
    assert "material" in text.lower()
    assert "texture" in text.lower()


def test_avatar_knight_metadata_fixture_is_valid_json():
    path = _fixture_path("avatar_knight_metadata.json")
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["asset_name"] == "knight_avatar"
    assert data["rig_type"] in {"R15", "R6"}
    assert isinstance(data["meshes"], list)
    assert isinstance(data["textures"], list)
    assert isinstance(data["accessories"], list)
