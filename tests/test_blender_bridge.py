import ast
from pathlib import Path

import pytest

from motionforge.blender_bridge import (
    generate_blender_import_script,
    write_blender_import_script,
)


def test_generated_script_is_valid_python_and_contains_bvh_import(tmp_path):
    script = generate_blender_import_script(
        tmp_path / "walk.bvh",
        tmp_path / "walk.blend",
    )
    ast.parse(script)
    assert "bpy.ops.import_anim.bvh" in script
    assert "bpy.ops.wm.save_as_mainfile" in script
    assert "CONFIG" in script


def test_character_import_and_guarded_action_assignment(tmp_path):
    script = generate_blender_import_script(
        tmp_path / "walk.bvh",
        tmp_path / "walk.blend",
        character_path=tmp_path / "hero.glb",
        target_armature="HeroRig",
    )
    ast.parse(script)
    assert "bpy.ops.import_scene.gltf" in script
    assert "target.animation_data.action = source_action" in script
    assert "Target rig is not name-compatible" in script


def test_rejects_unsupported_character_format(tmp_path):
    with pytest.raises(ValueError, match="Character must be"):
        generate_blender_import_script(
            tmp_path / "walk.bvh",
            tmp_path / "walk.blend",
            character_path=tmp_path / "hero.obj",
        )


def test_target_armature_requires_character(tmp_path):
    with pytest.raises(ValueError, match="requires character_path"):
        generate_blender_import_script(
            tmp_path / "walk.bvh",
            tmp_path / "walk.blend",
            target_armature="HeroRig",
        )


def test_write_script_creates_parent_directories(tmp_path):
    output = tmp_path / "generated" / "import_take.py"
    result = write_blender_import_script(
        tmp_path / "walk.bvh",
        output,
        tmp_path / "walk.blend",
    )
    assert result == output
    assert output.is_file()
    ast.parse(output.read_text(encoding="utf-8"))
