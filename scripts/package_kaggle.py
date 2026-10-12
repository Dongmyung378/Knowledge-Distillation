"""Package a credential-free, private Kaggle GPU experiment from exact sources.

Kaggle script kernels upload one code file. A deterministic, whitelisted source
archive is therefore embedded in that file and verified before extraction. The
kernel does not include local data, credentials, model files or personal notes.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

WORKER = r'''
import collections, importlib.metadata, json, os, platform, shutil, sys, time
from pathlib import Path

source = Path(__file__).resolve().parent
sys.path.insert(0, str(source))
contract = json.loads((source / "kaggle-contract.json").read_text())
output = Path("/kaggle/working/kd-results")
output.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import tensorflow as tf
from knowledge_distillation.data import build_split_manifest, file_sha256, save_manifest
from knowledge_distillation.experiments import run_experiment, source_inventory
from knowledge_distillation.training import require_kaggle_gpu

def write(value, name):
    (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

try:
    require_kaggle_gpu()
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The pinned GPU runtime requires Python 3.12")
    for line in (source / "requirements-runtime.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, wanted = line.split("==", 1)
        if importlib.metadata.version(name) != wanted:
            raise RuntimeError("Pinned runtime mismatch: " + name)
    if source_inventory(source) != contract["source_inventory"]:
        raise RuntimeError("Extracted experiment sources differ from the package contract")
    with tf.device("/GPU:0"):
        probe = tf.matmul(tf.ones((128, 128)), tf.ones((128, 128)))
        probe_sum = float(tf.reduce_sum(probe).numpy())
    if "GPU:0" not in probe.device or probe_sum != 2097152.0:
        raise RuntimeError("The real GPU computation did not match the preflight")
    gpu_report = {
        "schema_version": 1, "status": "passed", "python": platform.python_version(),
        "tensorflow": tf.__version__, "keras": importlib.metadata.version("keras"),
        "numpy": importlib.metadata.version("numpy"), "execution_device": probe.device,
        "gpu_details": [tf.config.experimental.get_device_details(x) for x in tf.config.list_physical_devices("GPU")],
        "matrix_sum": probe_sum, "source_inventory": source_inventory(source),
    }
    write(gpu_report, "gpu-runtime.json")
    print("KD GPU RUNTIME", json.dumps({k: gpu_report[k] for k in ("python", "tensorflow", "keras", "execution_device")}), flush=True)
    candidate_roots = [
        Path("/kaggle/input/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"),
        Path("/kaggle/input/gtsrb-german-traffic-sign"),
    ]
    candidates = [p for p in candidate_roots if (p / "Train.csv").is_file()]
    if len(candidates) != 1:
        raise RuntimeError("Expected the pinned GTSRB mirror at one canonical Kaggle input path")
    mounted = candidates[0]
    if file_sha256(mounted / "Train.csv") != contract["train_csv_sha256"]:
        raise RuntimeError("Kaggle input Train.csv differs from the verified dataset version")
    start = time.perf_counter()
    # Only original Train images are copied. Test.csv is not read in this job.
    import csv
    data = Path("/tmp/kd-gtsrb-training")
    data.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(mounted / "Train.csv", data / "Train.csv")
    with (data / "Train.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            from knowledge_distillation.data import safe_dataset_path
            original = safe_dataset_path(mounted, row["Path"])
            destination = safe_dataset_path(data, row["Path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    manifest = build_split_manifest(data, seed=378, ratios=(.7, .15, .15))
    if manifest["manifest_sha256"] != contract["manifest_sha256"]:
        raise RuntimeError("Kaggle image bytes or split assignment differ from the verified local manifest")
    hashes = collections.defaultdict(set)
    for record in manifest["records"]:
        hashes[record["sha256"]].add(record["split"])
    if any(len(names) > 1 for names in hashes.values()):
        raise RuntimeError("Exact duplicate images cross the declared split boundaries")
    write({"manifest_sha256": manifest["manifest_sha256"], "inventory_sha256": manifest["inventory_sha256"],
           "counts": manifest["counts"], "track_counts": manifest["track_counts"],
           "exact_image_sha_cross_split_duplicates": 0, "preparation_seconds": time.perf_counter() - start},
          "data-preflight.json")
    save_manifest(manifest, output / "splits.json")
    print("KD DATA VERIFIED", json.dumps(manifest["counts"]), flush=True)
    for stage in ("pilot", "confirmation", "freeze"):
        print("KD STAGE START", stage, flush=True)
        result = run_experiment(data, output, source / "configs/experiment-plan.json", stage,
                                manifest_path=output / "splits.json")
        print("KD STAGE COMPLETE", json.dumps({"stage": stage, "status": result["status"],
                    "decision": result.get("decision"), "deployment_kind": result.get("deployment_kind")}), flush=True)
    # Complete package versions support diagnosis without disclosing environment variables.
    packages = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    write({"python": platform.python_version(), "packages": packages}, "installed-packages.json")
    write({"status": "completed", "stages": ["pilot", "confirmation", "freeze"],
           "locked_holdout_evaluated": False, "official_test_evaluated": False}, "kernel-completion.json")
except Exception as error:
    write({"status": "failed", "type": type(error).__name__, "message": str(error)}, "kernel-failure.json")
    raise
'''


def source_files() -> list[Path]:
    files = sorted((ROOT / "knowledge_distillation").glob("*.py"))
    files += [ROOT / name for name in ("pyproject.toml", "requirements.txt", "requirements-runtime.txt")]
    files += sorted((ROOT / "scripts").glob("*.py"))
    files += [ROOT / "configs/experiment-plan.json"]
    # Documentation is context for this training package, not part of the
    # source-bound experiment inventory. Keep this explicit allowlist so that
    # private PDFs, local notes, model files and future documents stay excluded.
    files += [ROOT / name for name in ("README.md", "README.ko.md")]
    files += [ROOT / "docs" / (name + suffix) for name in
              ("architecture", "evaluation", "runtime", "training")
              for suffix in (".md", ".ko.md")]
    return sorted(set(files))


def package(output: Path, kernel_id: str, manifest_path: Path) -> dict:
    owner, sep, slug = kernel_id.partition("/")
    if not sep or not owner or not slug or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in slug):
        raise ValueError("Use a safe account/private-kernel-slug ID")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = source_files()
    helper = ROOT / "scripts/setup_kaggle_runtime.py"
    if helper not in files:
        raise FileNotFoundError("The pinned Kaggle runtime helper must be finalized before packaging")
    source_inventory = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in files if p.parent.name == "knowledge_distillation" or p.name in
                        {"pyproject.toml", "requirements.txt", "requirements-runtime.txt"}}
    contract = {"schema_version": 1, "source_inventory": source_inventory,
                "manifest_sha256": manifest["manifest_sha256"],
                "train_csv_sha256": "4e00ce71e4bdd9fc424d93f81ff82056c6b5cb9bbc48269741e628a134c4712b",
                "historical_exposure": manifest.get("historical_exposure"),
                "training_location": "Kaggle GPU only", "final_labels_accessed": False}
    embedded = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in files}
    embedded["kaggle-worker.py"] = WORKER.encode()
    embedded["kaggle-contract.json"] = (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(embedded.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    raw = buffer.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(raw).decode("ascii")
    outer = '''"""Private, hash-verified Kaggle-only GTSRB training kernel."""
import base64, hashlib, io, json, subprocess, sys, zipfile
from pathlib import Path, PurePosixPath
payload = base64.b64decode(__ENCODED__)
if hashlib.sha256(payload).hexdigest() != __SHA__:
    raise RuntimeError("Source package checksum mismatch")
target = Path("/tmp/kd-source")
target.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        relative = PurePosixPath(member.filename)
        destination = target.joinpath(*relative.parts)
        if relative.is_absolute() or ".." in relative.parts or not destination.resolve().is_relative_to(target):
            raise RuntimeError("Unsafe embedded archive path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))
subprocess.run([sys.executable, str(target / "scripts/setup_kaggle_runtime.py"),
                "--project-root", str(target), "--runtime", "/tmp/knowledge-distillation-runtime",
                "--metadata-file", "/kaggle/working/kd-results/kaggle-runtime.json",
                "--", str(target / "kaggle-worker.py")], cwd=target, check=True)
'''.replace("__ENCODED__", repr(encoded)).replace("__SHA__", repr(sha))
    output.mkdir(parents=True, exist_ok=False)
    (output / "run-experiment.py").write_text(outer, encoding="utf-8", newline="\n")
    metadata = {"id": kernel_id, "title": slug.replace("-", " "),
                "code_file": "run-experiment.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": ["meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    provenance = {"schema_version": 1, "kernel_id": kernel_id, "private": True,
                  "source_archive_sha256": sha, "source_archive_bytes": len(raw),
                  "embedded_files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(embedded.items())},
                  "manifest_sha256": manifest["manifest_sha256"],
                  "code_file_sha256": hashlib.sha256(outer.encode()).hexdigest(),
                  "training_location": "Kaggle GPU only"}
    (output / "package-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/split-manifest.json")
    args = parser.parse_args()
    result = package(args.output, args.kernel_id, args.manifest)
    print(json.dumps({k: result[k] for k in ("kernel_id", "source_archive_sha256", "source_archive_bytes", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
