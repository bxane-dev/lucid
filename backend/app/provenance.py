from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_source_manifest(
    paths: Iterable[Path],
    *,
    dataset_id: str,
    dataset_root: Path,
) -> dict:
    unique = sorted(
        {Path(path).resolve() for path in paths},
        key=lambda value: value.as_posix(),
    )
    if not unique:
        raise ValueError("Cannot build provenance for an empty source-file set.")

    files = []
    for path in unique:
        if not path.is_file():
            raise FileNotFoundError(f"Provenance source file missing: {path}")
        files.append(
            {
                "path": _display_path(path, dataset_root),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "files": files,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def manifest_path_for(prepared_path: Path) -> Path:
    prepared_path = Path(prepared_path)
    return prepared_path.with_name(prepared_path.name + ".manifest.json")


def write_manifest(manifest: dict, prepared_path: Path) -> Path:
    path = manifest_path_for(prepared_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
