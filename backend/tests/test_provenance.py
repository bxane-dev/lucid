import hashlib
import json

from app.provenance import (
    build_source_manifest,
    manifest_path_for,
    sha256_file,
    write_manifest,
)


def test_sha256_file_matches_known_digest(tmp_path):
    path = tmp_path / "recording.edf"
    path.write_bytes(b"real-public-eeg-bytes")
    assert sha256_file(path) == hashlib.sha256(
        b"real-public-eeg-bytes"
    ).hexdigest()


def test_manifest_is_deterministic_and_written(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    a = root / "sub-01_eeg.edf"
    b = root / "sub-01_events.tsv"
    a.write_bytes(b"eeg")
    b.write_text("onset\ttrial_type\n0\tyes\n", encoding="utf-8")

    first = build_source_manifest(
        [b, a],
        dataset_id="nm000113",
        dataset_root=root,
    )
    second = build_source_manifest(
        [a, b],
        dataset_id="nm000113",
        dataset_root=root,
    )

    assert first == second
    assert len(first["manifest_sha256"]) == 64
    assert [entry["path"] for entry in first["files"]] == [
        "sub-01_eeg.edf",
        "sub-01_events.tsv",
    ]

    prepared = tmp_path / "prepared" / "nm000113_words.npz"
    written = write_manifest(first, prepared)
    assert written == manifest_path_for(prepared)
    loaded = json.loads(written.read_text(encoding="utf-8"))
    assert loaded["manifest_sha256"] == first["manifest_sha256"]
