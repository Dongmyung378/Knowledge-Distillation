"""Prepare an isolated Python 3.12/TensorFlow 2.19 GPU runtime on Kaggle only.

Run this stdlib-only helper with Kaggle's system Python. Arguments after ``--``
are Python arguments executed by the managed virtual-environment interpreter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import uuid

UV_VERSION = "0.13.0"
PYTHON_VERSION = "3.12.15"
UV_ARCHIVE_URL = "https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_ARCHIVE_SHA256 = "1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154"
UV_ARCHIVE_BYTES = 17380567
UV_BINARY_SHA256 = "3e801df892439f5cde65d8b51ea69b69abff199ef37cc3cae4481be15d592df0"
UV_BINARY_BYTES = 39714040
UV_MEMBER = "uv-x86_64-unknown-linux-gnu/uv"
OWNER = {"schema_version": 1, "purpose": "knowledge-distillation-kaggle-runtime"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_kaggle() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The GPU bootstrap is for Linux x86-64 Kaggle kernels only.")
    if not Path("/kaggle/working").is_dir() or not (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").is_dir()
    ):
        raise RuntimeError("A real Kaggle kernel environment is required; no local GPU setup is performed.")


def scoped_path(target: Path, *, roots: tuple[Path, ...] = (Path("/tmp"), Path("/kaggle/working"))) -> Path:
    target = target.absolute()
    resolved = target.resolve()
    if not any(resolved != root.resolve() and resolved.is_relative_to(root.resolve()) for root in roots):
        raise ValueError("Kaggle runtime paths must be children of /tmp or /kaggle/working.")
    current = target
    while current != current.parent:
        if current.is_symlink():
            raise ValueError("Kaggle runtime paths cannot pass through symbolic links.")
        current = current.parent
    return resolved


def prepare_runtime_directory(runtime: Path) -> Path:
    runtime = scoped_path(runtime)
    marker = runtime / "runtime-owner.json"
    if runtime.exists():
        if marker.is_symlink() or not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != OWNER:
            raise ValueError("Existing runtime directory is not owned by this project; choose a new path.")
    else:
        runtime.mkdir(parents=True)
        marker.write_text(json.dumps(OWNER), encoding="utf-8")
    return runtime


def child_environment(runtime: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV", "TF_USE_LEGACY_KERAS", "TF_PLUGIN_PATH", "CUDA_HOME", "CUDA_PATH", "XLA_FLAGS", "TF_XLA_FLAGS"):
        env.pop(key, None)
    allowed_pip = {"PIP_CERT", "PIP_CLIENT_CERT", "PIP_PROXY", "PIP_TIMEOUT", "PIP_DEFAULT_TIMEOUT", "PIP_RETRIES"}
    for key in list(env):
        if (key.startswith("PIP_") and key not in allowed_pip) or key.startswith("UV_"):
            env.pop(key, None)
    temporary = scoped_path(runtime / "temp")
    temporary.mkdir(exist_ok=True)
    for directory in (runtime / "python", runtime / "uv-cache"):
        scoped_path(directory)
    env.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary),
               PYTHONNOUSERSITE="1", PYTHONUTF8="1", PIP_CONFIG_FILE=os.devnull,
               UV_PYTHON_INSTALL_DIR=str(runtime / "python"), UV_CACHE_DIR=str(runtime / "uv-cache"),
               TF_CPP_MIN_LOG_LEVEL="2", MPLBACKEND="Agg")
    # Preserve Kaggle's assigned CUDA_VISIBLE_DEVICES; do not change another
    # project's GPU allocation or force a CPU-only launcher environment.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        env.setdefault(key, "2")
    return env


def install_uv(runtime: Path) -> Path:
    tool_dir = scoped_path(runtime / "uv-tools" / UV_VERSION)
    tool_dir.mkdir(parents=True, exist_ok=True)
    binary = tool_dir / "uv"
    if binary.is_file() and not binary.is_symlink() and binary.stat().st_size == UV_BINARY_BYTES and sha256_file(binary) == UV_BINARY_SHA256:
        binary.chmod(0o700)
        return binary
    archive = tool_dir / "uv-download.tar.gz"
    temporary = tool_dir / ("uv-download-" + uuid.uuid4().hex + ".part")
    request = urllib.request.Request(UV_ARCHIVE_URL, headers={"User-Agent": "Knowledge-Distillation-Kaggle/2.0"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as target:
            if not response.geturl().startswith("https://"):
                raise ValueError("uv download redirected away from HTTPS")
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > UV_ARCHIVE_BYTES:
                    raise ValueError("uv archive exceeds the verified release size")
                digest.update(block)
                target.write(block)
        if size != UV_ARCHIVE_BYTES or digest.hexdigest() != UV_ARCHIVE_SHA256:
            raise ValueError("uv archive failed its official release checksum; it was not executed")
        temporary.replace(archive)
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember(UV_MEMBER)
            if not member.isfile() or member.size != UV_BINARY_BYTES:
                raise ValueError("uv is not the expected regular release executable")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError("Verified uv archive is missing its executable")
            contents = stream.read(UV_BINARY_BYTES + 1)
        if len(contents) != UV_BINARY_BYTES or hashlib.sha256(contents).hexdigest() != UV_BINARY_SHA256:
            raise ValueError("uv executable checksum mismatch; it was not executed")
        pending_binary = tool_dir / ("uv-" + uuid.uuid4().hex + ".part")
        pending_binary.write_bytes(contents)
        pending_binary.chmod(0o700)
        pending_binary.replace(binary)
        return binary
    finally:
        temporary.unlink(missing_ok=True)


PROBE = r'''
import importlib.metadata as m, json, pathlib, sys
problems = []
versions = {}
for requirement in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    requirement = requirement.strip()
    if not requirement or requirement.startswith("#"):
        continue
    name, wanted = requirement.split("==", 1)
    try:
        actual = m.version(name)
        versions[name] = actual
        if actual != wanted:
            problems.append(f"{name}: expected {wanted}, found {actual}")
    except m.PackageNotFoundError:
        problems.append(f"{name}: missing")
cuda_versions = {
    "gtsrb-knowledge-distillation": "2.0.0",
    "nvidia-cublas-cu12": "12.5.3.2", "nvidia-cuda-cupti-cu12": "12.5.82",
    "nvidia-cuda-nvcc-cu12": "12.5.82", "nvidia-cuda-nvrtc-cu12": "12.5.82",
    "nvidia-cuda-runtime-cu12": "12.5.82", "nvidia-cudnn-cu12": "9.3.0.75",
    "nvidia-cufft-cu12": "11.2.3.61", "nvidia-curand-cu12": "10.3.6.82",
    "nvidia-cusolver-cu12": "11.6.3.83", "nvidia-cusparse-cu12": "12.5.1.3",
    "nvidia-nccl-cu12": "2.23.4", "nvidia-nvjitlink-cu12": "12.5.82",
}
for package, wanted in cuda_versions.items():
    try:
        actual = m.version(package)
        versions[package] = actual
        if actual != wanted:
            problems.append(f"{package}: expected {wanted}, found {actual}")
    except m.PackageNotFoundError:
        problems.append(f"{package}: missing")
if sys.version_info[:3] != (3, 12, 15) or sys.prefix == sys.base_prefix:
    problems.append("The managed Python 3.12.15 virtual environment is required")
configuration = pathlib.Path(sys.prefix) / "pyvenv.cfg"
if not configuration.is_file() or "include-system-site-packages = true" in configuration.read_text().lower():
    problems.append("System site packages must be disabled")
print(json.dumps({"healthy": not problems, "python": sys.version.split()[0], "versions": versions, "problems": problems}))
sys.exit(bool(problems))
'''


def probe(python: Path, project: Path, env: dict[str, str]) -> dict:
    if not python.is_file():
        return {"healthy": False, "problems": ["Runtime interpreter is missing"]}
    try:
        completed = subprocess.run([str(python), "-I", "-c", PROBE, str(project / "requirements-runtime.txt")],
                                   env=env, cwd=project, capture_output=True, text=True, timeout=90)
        return json.loads(completed.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"healthy": False, "problems": ["Runtime dependency probe failed"]}


def cuda_environment(python: Path, env: dict[str, str]) -> dict[str, str]:
    result = subprocess.run([str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                            env=env, capture_output=True, text=True, check=True)
    site_packages = Path(result.stdout.strip())
    result_env = env.copy()
    libraries = sorted((site_packages / "nvidia").glob("*/lib"))
    inherited = env.get("LD_LIBRARY_PATH", "")
    result_env["LD_LIBRARY_PATH"] = ":".join([str(path) for path in libraries] + ([inherited] if inherited else []))
    nvcc_bin = site_packages / "nvidia" / "cuda_nvcc" / "bin"
    if nvcc_bin.is_dir():
        result_env["PATH"] = str(nvcc_bin) + os.pathsep + env.get("PATH", "")
    return result_env


GPU_PROBE = r'''
import json, platform, tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("TensorFlow 2.19 found no Kaggle GPU after isolated installation")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
with tf.device("/GPU:0"):
    value = tf.matmul(tf.constant([[1.0, 2.0]], dtype=tf.float32), tf.constant([[3.0], [4.0]], dtype=tf.float32))
if float(value.numpy()[0,0]) != 11.0 or "GPU:0" not in value.device:
    raise RuntimeError("The GPU smoke operation did not execute on GPU:0")
devices = []
for gpu in gpus:
    details = tf.config.experimental.get_device_details(gpu)
    devices.append({"name": str(details.get("device_name", "unknown")), "compute_capability": list(details.get("compute_capability", []))})
print(json.dumps({"gpu_count": len(gpus), "gpus": devices, "gpu_smoke_result": 11.0, "gpu_smoke_device": value.device.rsplit("/",1)[-1], "platform": platform.system(), "machine": platform.machine()}))
'''


def setup(project: Path, runtime: Path) -> tuple[Path, dict[str, str], dict]:
    require_kaggle()
    project = project.resolve(strict=True)
    if not (project / "requirements-runtime.txt").is_file() or not (project / "pyproject.toml").is_file():
        raise ValueError("Project source is missing pinned requirements or package metadata")
    runtime = prepare_runtime_directory(runtime)
    env = child_environment(runtime)
    uv = install_uv(runtime)
    subprocess.run([str(uv), "--no-config", "python", "install", "--no-bin", PYTHON_VERSION],
                   cwd=project, env=env, check=True)
    result = subprocess.run([str(uv), "--no-config", "python", "find", "--managed-python", PYTHON_VERSION],
                            cwd=project, env=env, capture_output=True, text=True, check=True)
    managed = Path(result.stdout.strip()).resolve(strict=True)
    if not managed.is_relative_to((runtime / "python").resolve()):
        raise ValueError("Managed Python unexpectedly resolved outside this task's runtime")
    venv = scoped_path(runtime / ".venv")
    python = venv / "bin" / "python"
    status = probe(python, project, env)
    if not status["healthy"]:
        if venv.exists():
            backup = runtime / ("venv-backup-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
            venv.rename(backup)
        subprocess.run([str(managed), "-I", "-m", "venv", str(venv)], cwd=project, env=env, check=True)
        pip = [str(python), "-I", "-m", "pip", "--disable-pip-version-check", "--no-cache-dir"]
        subprocess.run(pip + ["install", "--index-url", "https://pypi.org/simple", "-r", str(project / "requirements-runtime.txt"), "tensorflow[and-cuda]==2.19.0"],
                       cwd=project, env=env, check=True)
        subprocess.run(pip + ["install", "--no-deps", "--no-build-isolation", "-e", str(project)], cwd=project, env=env, check=True)
        subprocess.run([str(python), "-I", "-m", "pip", "check"], cwd=project, env=env, check=True)
        status = probe(python, project, env)
        if not status["healthy"]:
            raise RuntimeError("Isolated Kaggle dependency validation failed: " + "; ".join(status["problems"]))
    env = cuda_environment(python, env)
    gpu = subprocess.run([str(python), "-I", "-c", GPU_PROBE], cwd=project, env=env,
                         capture_output=True, text=True, timeout=180)
    if gpu.returncode:
        # Framework diagnostics contain no credentials. The final report omits
        # environment variables and paths from Kaggle's system installation.
        raise RuntimeError("Isolated TensorFlow GPU verification failed: " + gpu.stderr[-4000:])
    metadata = {"schema_version": 1, "purpose": OWNER["purpose"], "host_python": sys.version.split()[0],
                "python": status["python"], "versions": status["versions"], "isolated": True,
                "system_site_packages": False, "uv_version": UV_VERSION,
                "uv_archive_sha256": UV_ARCHIVE_SHA256, "uv_binary_sha256": UV_BINARY_SHA256,
                "requirements_sha256": sha256_file(project / "requirements-runtime.txt"),
                "local_gpu_used": False, **json.loads(gpu.stdout)}
    return python, env, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runtime", type=Path, default=Path("/tmp/knowledge-distillation-runtime"))
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("python_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        python, env, metadata = setup(args.project_root, args.runtime)
        if args.metadata_file:
            destination = scoped_path(args.metadata_file)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False), flush=True)
        command = args.python_arguments
        if command[:1] == ["--"]:
            command = command[1:]
        if args.setup_only or not command:
            return 0
        return subprocess.run([str(python), "-I", "-X", "utf8", *command], env=env, cwd=args.project_root).returncode
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError, subprocess.TimeoutExpired, tarfile.TarError) as error:
        print(json.dumps({"ready": False, "error": str(error)}), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
