"""MotionForge project manifest and safe asset/take registry.

The project file is intentionally provider-agnostic. Heavy model outputs are
referenced as files; the manifest never embeds generated meshes or video data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import uuid

MANIFEST_NAME = "motionforge.project.json"
SCHEMA_VERSION = 1


class ProjectError(ValueError):
    """Raised when a MotionForge project is invalid or unsafe."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectError("Project file paths must be non-empty strings.")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ProjectError(f"Unsafe project-relative path: {value!r}")
    if ":" in path.parts[0]:
        raise ProjectError(f"Drive-qualified paths are not allowed: {value!r}")
    return path.as_posix()


def _write_manifest(root: Path, data: dict) -> None:
    target = root / MANIFEST_NAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)


def create_project(directory: str | Path, name: str) -> Path:
    """Create a new project directory and return its manifest path."""
    if not isinstance(name, str) or not name.strip():
        raise ProjectError("Project name cannot be empty.")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / MANIFEST_NAME
    if manifest.exists():
        raise ProjectError(f"A MotionForge project already exists at {root}.")
    for folder in ("assets", "source", "takes", "exports", "cache"):
        (root / folder).mkdir(exist_ok=True)
    data = {
        "schema_version": SCHEMA_VERSION,
        "project_id": str(uuid.uuid4()),
        "name": name.strip(),
        "created_at": _now(),
        "updated_at": _now(),
        "assets": [],
        "takes": [],
    }
    _write_manifest(root, data)
    return manifest


def load_project(directory_or_manifest: str | Path) -> dict:
    """Load and minimally validate a project manifest."""
    candidate = Path(directory_or_manifest).expanduser()
    manifest = candidate / MANIFEST_NAME if candidate.is_dir() else candidate
    if manifest.name != MANIFEST_NAME or not manifest.is_file():
        raise ProjectError(f"Could not find {MANIFEST_NAME} at {manifest}.")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(f"Could not read project manifest: {exc}") from exc
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ProjectError(f"Unsupported project schema version: {data.get('schema_version')!r}")
    for key in ("project_id", "name", "assets", "takes"):
        if key not in data:
            raise ProjectError(f"Project manifest is missing required field: {key}")
    if not isinstance(data["assets"], list) or not isinstance(data["takes"], list):
        raise ProjectError("Project assets and takes must be lists.")
    return data


def add_asset(directory_or_manifest: str | Path, name: str, asset_type: str, file_path: str,
              metadata: dict | None = None) -> dict:
    """Register an asset using a project-relative path."""
    manifest = _manifest_path(directory_or_manifest)
    data = load_project(manifest)
    if not name.strip() or not asset_type.strip():
        raise ProjectError("Asset name and type are required.")
    relative = _safe_relative_path(file_path)
    if any(item["file"] == relative for item in data["assets"]):
        raise ProjectError(f"An asset already references {relative!r}.")
    record = {
        "asset_id": str(uuid.uuid4()),
        "name": name.strip(),
        "type": asset_type.strip().lower(),
        "file": relative,
        "metadata": metadata or {},
        "created_at": _now(),
    }
    data["assets"].append(record)
    data["updated_at"] = _now()
    _write_manifest(manifest.parent, data)
    return record


def add_take(directory_or_manifest: str | Path, name: str, source_video: str,
              motion_file: str | None = None, metadata: dict | None = None) -> dict:
    """Register a capture/animation take and its optional derived motion file."""
    manifest = _manifest_path(directory_or_manifest)
    data = load_project(manifest)
    if not name.strip():
        raise ProjectError("Take name is required.")
    source = _safe_relative_path(source_video)
    motion = _safe_relative_path(motion_file) if motion_file else None
    record = {
        "take_id": str(uuid.uuid4()),
        "name": name.strip(),
        "source_video": source,
        "motion_file": motion,
        "metadata": metadata or {},
        "created_at": _now(),
    }
    data["takes"].append(record)
    data["updated_at"] = _now()
    _write_manifest(manifest.parent, data)
    return record


def _manifest_path(directory_or_manifest: str | Path) -> Path:
    candidate = Path(directory_or_manifest).expanduser()
    return candidate / MANIFEST_NAME if candidate.is_dir() else candidate
