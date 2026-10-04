from __future__ import annotations

import argparse
from pathlib import Path
import sys
from urllib.parse import quote

BACKEND_DIR = Path(
    __file__
).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_DIR),
    )

import requests

from app.config import PUBLIC_DATA_DIR


REGISTRY = {
    "nm000113": {
        "version": "v1.0.0",
        "approx_size": "585 MB",
        "license": "CC-BY-4.0",
        "doi": "10.82901/nemar.nm000113",
    },
    "on003626": {
        "version": "v1.0.0",
        "approx_size": "24.6 GB",
        "license": "CC0",
        "doi": "10.82901/nemar.on003626",
    },
}

ROOT_METADATA = {
    "README.md",
    "dataset_description.json",
    "participants.tsv",
}


def list_directory(
    base_url: str,
    relative: str,
) -> list[dict]:
    path = quote(
        relative.strip("/"),
        safe="/",
    )
    url = (
        f"{base_url}/{path}/"
        if path
        else f"{base_url}/"
    )

    response = requests.get(
        url,
        params={"format": "json"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    if (
        payload.get("kind") != "directory"
        or "children" not in payload
    ):
        raise RuntimeError(
            "Unexpected NEMAR directory "
            f"response for {url}"
        )

    return payload["children"]


def download_file(
    base_url: str,
    relative: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):
        print(f"skip {relative}")
        return

    url = (
        f"{base_url}/"
        f"{quote(relative, safe='/')}"
    )
    tmp = destination.with_suffix(
        destination.suffix + ".part"
    )

    with requests.get(
        url,
        stream=True,
        timeout=120,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()

        with tmp.open("wb") as handle:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    handle.write(chunk)

    tmp.replace(destination)
    print(f"downloaded {relative}")


def download_tree(
    base_url: str,
    relative: str,
    destination_root: Path,
) -> None:
    for child in list_directory(
        base_url,
        relative,
    ):
        child_rel = (
            f"{relative.rstrip('/')}/"
            f"{child['name']}"
        ).lstrip("/")
        target = (
            destination_root / child_rel
        )

        if child["kind"] == "dir":
            download_tree(
                base_url,
                child_rel,
                destination_root,
            )
        elif child["kind"] == "file":
            download_file(
                base_url,
                child_rel,
                target,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download real public EEG "
            "directly from the official "
            "NEMAR archive. No generated "
            "EEG is created by this command."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(REGISTRY),
        default="nm000113",
    )
    parser.add_argument(
        "--subjects",
        default="all",
        help=(
            "Comma-separated BIDS subjects "
            "(for example sub-01,sub-02) "
            "or 'all'."
        ),
    )
    parser.add_argument(
        "--derivatives-only",
        action="store_true",
        help=(
            "Download only derivatives/<subject> "
            "instead of raw subject folders. "
            "Used for on003626 baseline and "
            "inner-speech epoch preparation."
        ),
    )
    args = parser.parse_args()

    meta = REGISTRY[args.dataset]
    version = meta["version"]
    base_url = (
        "https://data.nemar.org/"
        f"{args.dataset}/{version}"
    )
    destination = (
        PUBLIC_DATA_DIR / args.dataset
    )

    print(
        f"Dataset {args.dataset} "
        f"{version} | "
        f"{meta['approx_size']} "
        "full archive | "
        f"{meta['license']} | "
        f"DOI {meta['doi']}"
    )

    root_children = list_directory(
        base_url,
        "",
    )
    available_subjects = sorted(
        child["name"]
        for child in root_children
        if (
            child["kind"] == "dir"
            and child["name"].startswith(
                "sub-"
            )
        )
    )

    for child in root_children:
        if (
            child["kind"] == "file"
            and child["name"] in ROOT_METADATA
        ):
            download_file(
                base_url,
                child["name"],
                destination / child["name"],
            )

    if (
        args.subjects.strip().lower()
        == "all"
    ):
        selected = available_subjects
    else:
        selected = [
            value.strip()
            for value
            in args.subjects.split(",")
            if value.strip()
        ]
        unknown = sorted(
            set(selected)
            - set(available_subjects)
        )
        if unknown:
            raise SystemExit(
                "Unknown subjects: "
                + ", ".join(unknown)
            )

    if args.derivatives_only:
        derivatives_available = any(
            child["kind"] == "dir"
            and child["name"]
            == "derivatives"
            for child in root_children
        )
        if not derivatives_available:
            raise SystemExit(
                f"{args.dataset} does not "
                "publish a derivatives directory "
                "at this NEMAR version."
            )

        for subject in selected:
            download_tree(
                base_url,
                f"derivatives/{subject}",
                destination,
            )
    else:
        if (
            args.dataset == "on003626"
            and len(selected)
            == len(available_subjects)
        ):
            print(
                "Warning: on003626 raw data "
                "is about 24.6 GB in full."
            )

        for subject in selected:
            download_tree(
                base_url,
                subject,
                destination,
            )

    print(
        "Public EEG saved under: "
        f"{destination}"
    )


if __name__ == "__main__":
    main()
