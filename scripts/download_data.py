"""Install the CSV-referenced, case-preserving GTSRB dataset from a pinned ZIP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_artifacts import checked_path, download_file, relative_path, sha256_file, verify_file

INVENTORY_NAME = "dataset-inventory.json"
CSV_NAMES = {"Train.csv": "Train", "Test.csv": "Test", "Meta.csv": "Meta"}


def load_source(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("archive"), dict):
        raise ValueError("Expected a version-1 dataset descriptor with an archive")
    artifact = value["archive"]
    if not isinstance(artifact.get("bytes"), int) or isinstance(artifact["bytes"], bool) or artifact["bytes"] <= 0:
        raise ValueError("Dataset archive size is not valid")
    digest = artifact.get("sha256", "")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("Dataset archive SHA-256 is not valid")
    from urllib.parse import urlsplit
    if artifact.get("url") is not None:
        parsed = urlsplit(artifact["url"])
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Dataset archive URL must be credential-free HTTPS")
    return value


def _inventory_digest(entries: list[dict]) -> str:
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selected_members(bundle: zipfile.ZipFile, source: dict) -> tuple[list[str], dict[str, int]]:
    members = {}
    for member in bundle.infolist():
        path = member.filename.rstrip("/") if member.is_dir() else member.filename
        relative_path(path)
        if member.filename in members:
            raise ValueError("The dataset ZIP contains duplicate exact member names")
        members[member.filename] = member
    selected = list(CSV_NAMES)
    counts = {}
    selected_casefold = {name.casefold() for name in selected}
    for csv_name, prefix in CSV_NAMES.items():
        if csv_name not in members or members[csv_name].file_size > 16 * 1024 * 1024:
            raise ValueError(f"Missing or oversized root CSV: {csv_name}")
        text = bundle.read(csv_name).decode("utf-8-sig")
        rows = csv.DictReader(io.StringIO(text))
        if not rows.fieldnames or "Path" not in rows.fieldnames:
            raise ValueError(f"Dataset CSV is missing its Path column: {csv_name}")
        count = 0
        for row in rows:
            name = row["Path"]
            relative_path(name)
            if not name.startswith(prefix + "/") or name not in members or members[name].is_dir():
                raise ValueError(f"CSV path lacks its exact case-preserving archive member: {name}")
            if name.casefold() in selected_casefold:
                raise ValueError("CSV-referenced image paths collide or are duplicated")
            if "ClassId" in row and not 0 <= int(row["ClassId"]) < 43:
                raise ValueError("GTSRB class IDs must be in 0..42")
            selected_casefold.add(name.casefold())
            selected.append(name)
            count += 1
        counts[prefix] = count
    expected_counts = source.get("expected_counts")
    if expected_counts is not None and counts != expected_counts:
        raise ValueError("Dataset CSV counts differ from the published source descriptor")
    return sorted(selected), counts


def extract_canonical_dataset(archive: Path, output: Path, source: dict) -> dict:
    """Extract only exact CSV references; ignored case variants are never written."""
    output.mkdir(parents=True, exist_ok=False)
    inventory = []
    with zipfile.ZipFile(archive) as bundle:
        selected, counts = _selected_members(bundle, source)
        total = sum(bundle.getinfo(name).file_size for name in selected)
        if total > 2 * 1024**3:
            raise ValueError("Canonical dataset exceeds the supported extracted-size bound")
        for name in selected:
            member = bundle.getinfo(name)
            if member.file_size <= 0 or member.file_size > 25 * 1024**2:
                raise ValueError("Canonical dataset contains an empty or oversized member")
            import stat
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Dataset ZIP symbolic links are not supported")
            destination = checked_path(output, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with bundle.open(member) as stream, destination.open("xb") as target:
                while block := stream.read(1024 * 1024):
                    count += len(block)
                    if count > member.file_size:
                        raise ValueError("Dataset ZIP member exceeded its declared size")
                    digest.update(block)
                    target.write(block)
            if count != member.file_size:
                raise ValueError("Dataset ZIP member size differs after extraction")
            inventory.append({"path": name, "bytes": count, "sha256": digest.hexdigest()})
    digest = _inventory_digest(inventory)
    if source.get("canonical_inventory_sha256") is not None and digest != source["canonical_inventory_sha256"]:
        raise ValueError("Canonical dataset inventory differs from the published hash")
    report = {"schema_version": 1, "archive_sha256": source["archive"]["sha256"],
              "canonical_inventory_sha256": digest, "counts": counts, "files": inventory,
              "extraction_policy": "exact_root_csv_references_only; case-variant_duplicates_ignored"}
    (output / INVENTORY_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def verify_dataset(source_manifest: Path, output: Path) -> dict:
    source = load_source(source_manifest)
    inventory_path = checked_path(output, INVENTORY_NAME)
    report = json.loads(inventory_path.read_text(encoding="utf-8"))
    if report.get("archive_sha256") != source["archive"]["sha256"]:
        raise ValueError("Installed dataset came from a different archive")
    inventory = report["files"]
    digest = _inventory_digest(inventory)
    if digest != report.get("canonical_inventory_sha256"):
        raise ValueError("Installed dataset inventory hash differs")
    if source.get("canonical_inventory_sha256") is not None and digest != source["canonical_inventory_sha256"]:
        raise ValueError("Installed dataset inventory differs from the published source")
    if source.get("expected_counts") is not None and report.get("counts") != source["expected_counts"]:
        raise ValueError("Installed dataset counts differ from the published source")
    expected = {INVENTORY_NAME}
    for entry in inventory:
        path = checked_path(output, entry["path"])
        verify_file(path, entry)
        expected.add(entry["path"])
    actual = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("Installed dataset contains a symbolic link")
        if path.is_file():
            actual.add(path.relative_to(output).as_posix())
    if actual != expected:
        raise ValueError("Installed dataset has unexpected or missing files")
    return {"verified": True, "dataset_root": str(output.resolve()), "counts": report["counts"],
            "files": len(inventory), "archive_sha256": report["archive_sha256"], "canonical_inventory_sha256": digest}


def install_dataset(source_manifest: Path, output: Path, *, archive: Path | None = None) -> dict:
    source = load_source(source_manifest)
    output = output.absolute()
    if output.is_symlink() or output == output.parent:
        raise ValueError("Dataset output must be a regular directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return verify_dataset(source_manifest, output)
    lock = output.parent / ("." + output.name + ".install.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Another dataset installation is running") from error
    staging = None
    try:
        os.close(lock_fd)
        staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".staging-", dir=output.parent))
        if archive is None:
            archive = staging / "gtsrb.zip"
            artifact = {"name": "GTSRB dataset", "path": "gtsrb.zip", **source["archive"]}
            download_file(artifact, archive)
        else:
            verify_file(archive, source["archive"])
        extracted = staging / "installed"
        extract_canonical_dataset(archive, extracted, source)
        result = verify_dataset(source_manifest, extracted)
        if output.exists():
            raise ValueError("Dataset output appeared during installation; it was not overwritten")
        extracted.rename(output)
        result["dataset_root"] = str(output.resolve())
        return result
    finally:
        if staging is not None and staging.parent.resolve() == output.parent.resolve() and staging.name.startswith("." + output.name + ".staging-"):
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("reports/dataset-download.json"))
    parser.add_argument("--output", type=Path, default=Path(".local/data/gtsrb-v1"))
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_dataset(args.source_manifest, args.output) if args.verify_only else install_dataset(args.source_manifest, args.output, archive=args.archive)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({"verified": False, "error": str(error)}))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
