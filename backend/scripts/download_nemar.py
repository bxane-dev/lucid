from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import quote

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

ROOT_METADATA = {"README.md", "dataset_description.json", "participants.tsv"}


def list_directory(base_url: str, relative: str) -> list[dict]:
    path = quote(relative.strip("/"), safe="/")
    url = f"{base_url}/{path}/" if path else f"{base_url}/"
    response = requests.get(url, params={"format": "json"}, timeout=60)
    response.raise_for_status()
    return response.json()["children"]


def download_file(base_url: str, relative: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"skip {relative}")
        return

    url = f"{base_url}/{quote(relative, safe='/')}"
    tmp = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp.replace(destination)
    print(f"downloaded {relative}")


def download_tree(base_url: str, relative: str, destination_root: Path) -> None:
    for child in list_directory(base_url, relative):
        child_rel = f"{relative.rstrip('/')}/{child['name']}".lstrip("/")
        target = destination_root / child_rel
        if child["kind"] == "dir":
            download_tree(base_url, child_rel, destination_root)
        elif child["kind"] == "file":
            download_file(base_url, child_rel, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real public EEG directly from the official NEMAR archive."
    )
    parser.add_argument("--dataset", choices=sorted(REGISTRY), default="nm000113")
    parser.add_argument(
        "--subjects",
        default="all",
        help="Comma-separated BIDS subjects (e.g. sub-01,sub-02) or 'all'.",
    )
    args = parser.parse_args()

    meta = REGISTRY[args.dataset]
    version = meta["version"]
    base_url = f"https://data.nemar.org/{args.dataset}/{version}"
    destination = PUBLIC_DATA_DIR / args.dataset

    print(
        f"Dataset {args.dataset} {version} | {meta['approx_size']} full archive | "
        f"{meta['license']} | DOI {meta['doi']}"
    )

    root_children = list_directory(base_url, "")
    available_subjects = sorted(
        c["name"] for c in root_children
        if c["kind"] == "dir" and c["name"].startswith("sub-")
    )

    for child in root_children:
        if child["kind"] == "file" and child["name"] in ROOT_METADATA:
            download_file(base_url, child["name"], destination / child["name"])

    if args.subjects.strip().lower() == "all":
        selected = available_subjects
    else:
        selected = [x.strip() for x in args.subjects.split(",") if x.strip()]
        unknown = sorted(set(selected) - set(available_subjects))
        if unknown:
            raise SystemExit(f"Unknown subjects: {', '.join(unknown)}")

    if args.dataset == "on003626" and len(selected) == len(available_subjects):
        print("Warning: on003626 is about 24.6 GB in full.")

    for subject in selected:
        download_tree(base_url, subject, destination)

    print(f"Public EEG saved under: {destination}")


if __name__ == "__main__":
    main()
