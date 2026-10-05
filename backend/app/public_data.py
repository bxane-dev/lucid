from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import requests

from .config import PREPARED_DATA_DIR, PUBLIC_DATA_DIR


Progress = Callable[[dict], None]


DATASETS = {
    "nm000113": {
        "id": "nm000113",
        "title": "2020 BCI Competition Track 3",
        "description": "Imagined-speech EEG with five recorded commands.",
        "version": "v1.0.0",
        "license": "CC-BY-4.0",
        "doi": "10.82901/nemar.nm000113",
        "approx_size": "585 MB",
        "tasks": ["words"],
        "labels": ["Hello", "Help me", "Stop", "Thank you", "Yes"],
        "recommended_subjects": ["sub-01", "sub-02", "sub-03"],
        "derivatives_only": False,
    },
    "on003626": {
        "id": "on003626",
        "title": "Nieto Inner Speech",
        "description": "Inner-speech direction trials plus published resting baseline derivatives.",
        "version": "v1.0.0",
        "license": "CC0",
        "doi": "10.82901/nemar.on003626",
        "approx_size": "24.6 GB full archive",
        "tasks": ["words", "state"],
        "labels": ["Up", "Down", "Right", "Left", "Rest", "Imagined speech"],
        "recommended_subjects": ["sub-01", "sub-02", "sub-03"],
        "derivatives_only": True,
    },
}

ROOT_METADATA = {
    "README.md",
    "dataset_description.json",
    "participants.tsv",
}


@dataclass(frozen=True)
class RemoteFile:
    relative: str
    bytes: int | None = None


def _child_size(child: dict) -> int | None:
    for key in ("size", "bytes", "size_bytes"):
        value = child.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def registry_entry(dataset_id: str) -> dict:
    try:
        return DATASETS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported public dataset: {dataset_id}") from exc


def base_url(dataset_id: str) -> str:
    meta = registry_entry(dataset_id)
    return f"https://data.nemar.org/{dataset_id}/{meta['version']}"


def list_directory(dataset_id: str, relative: str) -> list[dict]:
    path = quote(relative.strip("/"), safe="/")
    url = f"{base_url(dataset_id)}/{path}/" if path else f"{base_url(dataset_id)}/"
    response = requests.get(url, params={"format": "json"}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if payload.get("kind") != "directory" or "children" not in payload:
        raise RuntimeError(f"Unexpected NEMAR directory response for {url}")
    return payload["children"]


def available_subjects(dataset_id: str) -> list[str]:
    return sorted(
        child["name"]
        for child in list_directory(dataset_id, "")
        if child.get("kind") == "dir"
        and child.get("name", "").startswith("sub-")
    )


def _walk(dataset_id: str, relative: str) -> list[RemoteFile]:
    files: list[RemoteFile] = []
    for child in list_directory(dataset_id, relative):
        child_rel = f"{relative.rstrip('/')}/{child['name']}".lstrip("/")
        if child["kind"] == "dir":
            files.extend(_walk(dataset_id, child_rel))
        elif child["kind"] == "file":
            files.append(
                RemoteFile(
                    relative=child_rel,
                    bytes=_child_size(child),
                )
            )
    return files


def plan_download(
    dataset_id: str,
    *,
    subjects: list[str] | None = None,
    derivatives_only: bool | None = None,
    progress: Progress | None = None,
) -> tuple[list[RemoteFile], list[str], bool]:
    meta = registry_entry(dataset_id)
    use_derivatives = (
        meta["derivatives_only"]
        if derivatives_only is None
        else derivatives_only
    )

    root = list_directory(dataset_id, "")
    available = sorted(
        child["name"]
        for child in root
        if child.get("kind") == "dir"
        and child.get("name", "").startswith("sub-")
    )
    selected = available if subjects is None else sorted(set(subjects))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError("Unknown subjects: " + ", ".join(unknown))
    if len(selected) < 3:
        raise ValueError(
            "Lucid requires at least 3 participants for participant-held-out "
            "train/validation/test splitting."
        )

    files = [
        RemoteFile(child["name"], _child_size(child))
        for child in root
        if child.get("kind") == "file" and child.get("name") in ROOT_METADATA
    ]

    if progress:
        progress({"phase": "discovering", "message": "Discovering public recording files."})

    for subject in selected:
        relative = (
            f"derivatives/{subject}"
            if use_derivatives
            else subject
        )
        files.extend(_walk(dataset_id, relative))

    return files, selected, use_derivatives


def download_file(
    dataset_id: str,
    remote: RemoteFile,
    destination_root: Path,
    *,
    on_bytes: Callable[[int], None] | None = None,
) -> bool:
    destination = destination_root / remote.relative
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        size = destination.stat().st_size
        if size > 0 and (remote.bytes is None or size == remote.bytes):
            return False
        destination.unlink()

    url = f"{base_url(dataset_id)}/{quote(remote.relative, safe='/')}"
    tmp = destination.with_suffix(destination.suffix + ".part")
    offset = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}

    try:
        with requests.get(
            url,
            stream=True,
            timeout=120,
            allow_redirects=True,
            headers=headers,
        ) as response:
            if offset and response.status_code == 416:
                if remote.bytes is not None and offset == remote.bytes:
                    tmp.replace(destination)
                    return True
                tmp.unlink(missing_ok=True)
                return download_file(
                    dataset_id,
                    remote,
                    destination_root,
                    on_bytes=on_bytes,
                )

            response.raise_for_status()

            resumed = bool(offset and response.status_code == 206)
            mode = "ab" if resumed else "wb"
            if not resumed:
                offset = 0

            with tmp.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        if on_bytes:
                            on_bytes(len(chunk))

        final_size = tmp.stat().st_size
        if final_size <= 0:
            raise RuntimeError(f"Downloaded empty public EEG file: {remote.relative}")
        if remote.bytes is not None and final_size != remote.bytes:
            raise RuntimeError(
                f"Public file size mismatch for {remote.relative}: "
                f"expected {remote.bytes}, received {final_size}."
            )

        tmp.replace(destination)
    except Exception:
        # Keep a non-empty .part file so a later attempt can resume it.
        if tmp.exists() and tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
        raise

    return True


def download_dataset(
    dataset_id: str,
    *,
    subjects: list[str] | None = None,
    derivatives_only: bool | None = None,
    progress: Progress | None = None,
) -> dict:
    files, selected, use_derivatives = plan_download(
        dataset_id,
        subjects=subjects,
        derivatives_only=derivatives_only,
        progress=progress,
    )
    destination = PUBLIC_DATA_DIR / dataset_id
    total_files = len(files)
    expected_bytes = sum(item.bytes or 0 for item in files)
    completed_files = 0
    downloaded_bytes = 0

    def add_bytes(value: int) -> None:
        nonlocal downloaded_bytes
        downloaded_bytes += value
        if progress:
            progress(
                {
                    "phase": "downloading",
                    "files_done": completed_files,
                    "files_total": total_files,
                    "bytes_downloaded": downloaded_bytes,
                    "bytes_expected": expected_bytes or None,
                }
            )

    for remote in files:
        download_file(
            dataset_id,
            remote,
            destination,
            on_bytes=add_bytes,
        )
        completed_files += 1
        if progress:
            progress(
                {
                    "phase": "downloading",
                    "files_done": completed_files,
                    "files_total": total_files,
                    "bytes_downloaded": downloaded_bytes,
                    "bytes_expected": expected_bytes or None,
                    "message": remote.relative,
                }
            )

    return {
        "dataset_id": dataset_id,
        "subjects": selected,
        "derivatives_only": use_derivatives,
        "files": total_files,
        "downloaded_bytes": downloaded_bytes,
        "destination": str(destination),
    }


def local_dataset_status(dataset_id: str) -> dict:
    meta = registry_entry(dataset_id)
    root = PUBLIC_DATA_DIR / dataset_id
    files = [path for path in root.rglob("*") if path.is_file()] if root.exists() else []
    partials = [path for path in files if path.suffix == ".part"]
    complete_files = [path for path in files if path.suffix != ".part"]
    bytes_on_disk = sum(path.stat().st_size for path in complete_files)

    subject_root = (
        root / "derivatives"
        if meta["derivatives_only"]
        else root
    )
    local_subjects = (
        sorted(
            path.name
            for path in subject_root.glob("sub-*")
            if path.is_dir()
        )
        if subject_root.exists()
        else []
    )

    word_archive = PREPARED_DATA_DIR / f"{dataset_id}_words.npz"
    state_archive = PREPARED_DATA_DIR / f"{dataset_id}_state.npz"

    return {
        **meta,
        "downloaded": len(local_subjects) >= 3 and not partials,
        "local_subjects": local_subjects,
        "downloaded_files": len(complete_files),
        "bytes_on_disk": bytes_on_disk,
        "partial_files": len(partials),
        "word_prepared": word_archive.exists(),
        "state_prepared": state_archive.exists(),
    }
