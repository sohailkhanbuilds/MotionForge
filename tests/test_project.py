import json
import pytest

from motionforge.project import (
    MANIFEST_NAME, ProjectError, add_asset, add_take, create_project, load_project,
)


def test_create_and_load_project(tmp_path):
    manifest = create_project(tmp_path / "Demo", "My Animation")
    data = load_project(manifest)
    assert manifest.name == MANIFEST_NAME
    assert data["name"] == "My Animation"
    assert data["schema_version"] == 1
    assert (manifest.parent / "assets").is_dir()
    assert (manifest.parent / "source").is_dir()


def test_register_asset_and_take(tmp_path):
    manifest = create_project(tmp_path / "Demo", "Demo")
    asset = add_asset(manifest, "Hero", "character", "assets/hero.glb", {"format": "glb"})
    take = add_take(manifest, "Walk", "source/walk.mp4", "takes/walk.bvh")
    data = load_project(manifest)
    assert data["assets"][0]["asset_id"] == asset["asset_id"]
    assert data["takes"][0]["take_id"] == take["take_id"]
    assert data["takes"][0]["motion_file"] == "takes/walk.bvh"


@pytest.mark.parametrize("path", ["../outside.glb", "/absolute.glb", "C:/outside.glb", ""])
def test_reject_unsafe_asset_paths(tmp_path, path):
    manifest = create_project(tmp_path / "Demo", "Demo")
    with pytest.raises(ProjectError):
        add_asset(manifest, "Bad", "character", path)


def test_prevent_duplicate_asset_file(tmp_path):
    manifest = create_project(tmp_path / "Demo", "Demo")
    add_asset(manifest, "Hero", "character", "assets/hero.glb")
    with pytest.raises(ProjectError):
        add_asset(manifest, "Hero copy", "character", "assets/hero.glb")


def test_prevent_overwriting_existing_project(tmp_path):
    manifest = create_project(tmp_path / "Demo", "Demo")
    with pytest.raises(ProjectError):
        create_project(manifest.parent, "Second")

from motionforge.project_cli import main as project_main


def test_project_cli_init_info_and_registration(tmp_path, capsys):
    project_dir = tmp_path / "CliDemo"
    assert project_main(["init", str(project_dir), "--name", "CLI Demo"]) == 0
    assert project_main([
        "add-asset", str(project_dir), "--name", "Hero",
        "--type", "character", "--file", "assets/hero.glb",
    ]) == 0
    assert project_main([
        "add-take", str(project_dir), "--name", "Walk",
        "--video", "source/walk.mp4", "--motion", "takes/walk.bvh",
    ]) == 0
    assert project_main(["info", str(project_dir)]) == 0
    output = capsys.readouterr().out
    assert "CLI Demo" in output
    assert "Hero" in output
    assert "Walk" in output
