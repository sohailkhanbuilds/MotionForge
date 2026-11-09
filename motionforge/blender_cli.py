"""Command-line helpers for optional Blender integration."""
from __future__ import annotations

import argparse
from pathlib import Path

from motionforge.blender_bridge import write_blender_import_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motionforge blender-script",
        description="Generate a Blender-side script to import a MotionForge BVH take.",
    )
    parser.add_argument("--bvh", required=True, help="Input MotionForge .bvh file")
    parser.add_argument("--script", required=True, help="Output Blender .py script")
    parser.add_argument("--blend-output", required=True, help="Output .blend project path")
    parser.add_argument("--character", help="Optional character .fbx, .glb or .gltf asset")
    parser.add_argument(
        "--target-armature",
        help="Optional imported Blender armature object name",
    )
    parser.add_argument(
        "--preview-scene",
        action="store_true",
        help="Add a basic camera, floor and three-point area lighting setup in Blender",
    )
    parser.add_argument(
        "--bone-map",
        help="Optional JSON object mapping BVH source bone names to target rig bone names",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = write_blender_import_script(
        bvh_path=args.bvh,
        script_path=args.script,
        blend_output=args.blend_output,
        character_path=args.character,
        target_armature=args.target_armature,
        bone_map_path=args.bone_map,
        preview_scene=args.preview_scene,
    )
    print(f"Blender script written: {path}")
    print("Run it with: blender --background --python " + str(path))
    print("Blender must be installed; runtime import is not performed by this command.")
    return 0
