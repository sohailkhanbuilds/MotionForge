"""Project-management commands for MotionForge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from motionforge.project import (
    ProjectError, add_asset, add_take, create_project, load_project,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="motionforge project", description="Manage MotionForge projects.")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create a new project")
    init.add_argument("directory", help="Project directory")
    init.add_argument("--name", required=True, help="Project display name")

    info = commands.add_parser("info", help="Show project manifest")
    info.add_argument("project", help="Project directory or manifest path")

    asset = commands.add_parser("add-asset", help="Register an existing or planned asset file")
    asset.add_argument("project")
    asset.add_argument("--name", required=True)
    asset.add_argument("--type", required=True, dest="asset_type")
    asset.add_argument("--file", required=True, help="Project-relative asset path")

    take = commands.add_parser("add-take", help="Register a source video and optional motion file")
    take.add_argument("project")
    take.add_argument("--name", required=True)
    take.add_argument("--video", required=True, help="Project-relative source video path")
    take.add_argument("--motion", help="Project-relative derived motion path")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            manifest = create_project(args.directory, args.name)
            print(f"Created MotionForge project: {manifest}")
        elif args.command == "info":
            print(json.dumps(load_project(args.project), indent=2))
        elif args.command == "add-asset":
            record = add_asset(args.project, args.name, args.asset_type, args.file)
            print(json.dumps(record, indent=2))
        elif args.command == "add-take":
            record = add_take(args.project, args.name, args.video, args.motion)
            print(json.dumps(record, indent=2))
        return 0
    except (ProjectError, OSError) as exc:
        parser.exit(2, f"MotionForge project error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
