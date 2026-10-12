"""Download hash-pinned model artifacts without executing remote code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from urllib.parse import urlsplit
import urllib.request
import zipfile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Artifact paths must be non-empty portable relative paths")
    parts = PurePosixPath(value)
    if parts.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"Unsafe artifact path: {value!r}")
    return Path(*parts.parts)


def checked_path(root: Path, value: str) -> Path:
    target = root / relative_path(value)
    current = root
    if current.is_symlink():
        raise ValueError("Artifact directories cannot be symbolic links")
    for part in target.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Artifact paths cannot contain symbolic links")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes the output directory")
    return target


def _size_hash(entry: dict) -> None:
    size = entry.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Artifact byte counts must be positive integers")
    if not isinstance(entry.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise ValueError("A lowercase SHA-256 hash is required for every artifact")


def load_descriptor(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("artifacts"), list) or not document["artifacts"]:
        raise ValueError("Expected a version-1 descriptor with at least one artifact")
    paths = set()
    inventory_paths = set()
    for artifact in document["artifacts"]:
        _size_hash(artifact)
        relative_path(artifact["path"])
        normalized = artifact["path"].casefold()
        if normalized in paths:
            raise ValueError("Duplicate artifact paths")
        paths.add(normalized)
        if artifact.get("kind", "file") not in {"file", "zip"}:
            raise ValueError("Supported artifact kinds are file and zip")
        url = artifact.get("url")
        if url is not None:
            parsed = urlsplit(url)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Artifact URLs must use HTTPS without embedded credentials")
        entries = artifact.get("inventory", []) if artifact.get("kind", "file") == "zip" else [artifact]
        if not entries:
            raise ValueError("ZIP artifacts require an exact extracted-file inventory")
        local_paths = set()
        for entry in entries:
            _size_hash(entry)
            relative_path(entry["path"])
            normalized = entry["path"].casefold()
            if normalized in local_paths:
                raise ValueError("Duplicate ZIP inventory paths")
            local_paths.add(normalized)
            complete = (artifact["path"] + "/" + entry["path"]) if artifact.get("kind", "file") == "zip" else artifact["path"]
            if complete.casefold() in inventory_paths:
                raise ValueError("Artifact inventories overlap")
            inventory_paths.add(complete.casefold())
    return document


def verify_file(path: Path, entry: dict) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular artifact file: {path}")
    if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
        raise ValueError(f"Artifact size/SHA-256 mismatch: {path.name}")


def verify_artifacts(descriptor: Path | dict, output: Path) -> dict:
    document = load_descriptor(descriptor) if isinstance(descriptor, Path) else descriptor
    expected = set()
    for artifact in document["artifacts"]:
        base = checked_path(output, artifact["path"])
        if artifact.get("kind", "file") == "file":
            verify_file(base, artifact)
            expected.add(base.relative_to(output).as_posix())
        else:
            for entry in artifact["inventory"]:
                target = checked_path(base, entry["path"])
                verify_file(target, entry)
                expected.add(target.relative_to(output).as_posix())
    if output.is_dir():
        actual = set()
        for path in output.rglob("*"):
            if path.is_symlink():
                raise ValueError("Installed artifact inventory contains a symbolic link")
            if path.is_file():
                actual.add(path.relative_to(output).as_posix())
        if actual != expected:
            raise ValueError("Installed artifact inventory differs from the descriptor")
    return {"verified": True, "files": len(expected), "bytes": sum((output / path).stat().st_size for path in expected)}


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        parsed = urlsplit(new_url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise ValueError("Download redirect is not a credential-free HTTPS URL")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def download_file(entry: dict, destination: Path) -> None:
    if not entry.get("url"):
        raise ValueError(f"No verified public download URL is available for {entry.get('name', entry['path'])}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(HTTPSRedirects())
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "GTSRB-Knowledge-Distillation/2.0"})
    count = 0
    digest = hashlib.sha256()
    with opener.open(request, timeout=60) as response, destination.open("xb") as stream:
        while block := response.read(1024 * 1024):
            count += len(block)
            if count > entry["bytes"]:
                raise ValueError("Downloaded artifact exceeds its declared size")
            digest.update(block)
            stream.write(block)
    if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError("Downloaded artifact failed its size/SHA-256 check")


def extract_verified_zip(archive: Path, output: Path, inventory: list[dict]) -> None:
    expected = {entry["path"]: entry for entry in inventory}
    seen = set()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                relative_path(member.filename.rstrip("/"))
                continue
            relative_path(member.filename)
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("ZIP symbolic links are not supported")
            if member.filename in seen or member.filename not in expected:
                raise ValueError("ZIP contains duplicate or undeclared files")
            seen.add(member.filename)
            entry = expected[member.filename]
            if member.file_size != entry["bytes"]:
                raise ValueError("ZIP file size differs from the inventory")
            target = checked_path(output, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with bundle.open(member) as source, target.open("xb") as destination:
                while block := source.read(1024 * 1024):
                    total += len(block)
                    if total > entry["bytes"]:
                        raise ValueError("ZIP member exceeds its declared size")
                    destination.write(block)
            verify_file(target, entry)
    if seen != set(expected):
        raise ValueError("ZIP is missing files from the declared inventory")


def install_artifacts(descriptor: Path, output: Path) -> dict:
    document = load_descriptor(descriptor)
    output = output.absolute()
    if output.is_symlink() or output == output.parent:
        raise ValueError("Output must be a regular artifact directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        return verify_artifacts(document, output)
    lock = output.parent / ("." + output.name + ".install.lock")
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise ValueError("Another artifact installation is running; retry after it completes") from error
    staging = None
    try:
        os.close(lock_fd)
        staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".staging-", dir=output.parent))
        extracted = staging / "installed"
        extracted.mkdir()
        for index, artifact in enumerate(document["artifacts"]):
            downloaded = staging / f"download-{index}"
            download_file(artifact, downloaded)
            destination = checked_path(extracted, artifact["path"])
            if artifact.get("kind", "file") == "zip":
                extract_verified_zip(downloaded, destination, artifact["inventory"])
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                downloaded.rename(destination)
        result = verify_artifacts(document, extracted)
        if output.exists():
            raise ValueError("Artifact output appeared while downloading; no existing files were overwritten")
        extracted.rename(output)
        return result
    finally:
        if staging is not None and staging.parent.resolve() == output.parent.resolve() and staging.name.startswith("." + output.name + ".staging-"):
            shutil.rmtree(staging)
        lock.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("reports/artifact-download.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/runtime"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_artifacts(args.manifest, args.output) if args.verify_only else install_artifacts(args.manifest, args.output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(json.dumps({"verified": False, "error": str(error)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
