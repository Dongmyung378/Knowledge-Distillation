"""Create a pinned, isolated Python 3.12 runtime inside this project."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import importlib.metadata as m, json, pathlib, sys
problems = []
for requirement in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    requirement = requirement.strip()
    if not requirement or requirement.startswith("#"):
        continue
    package, wanted = requirement.split("==", 1)
    try:
        actual = m.version(package)
        if actual != wanted:
            problems.append(f"{package}: wanted {wanted}, found {actual}")
    except m.PackageNotFoundError:
        problems.append(f"{package}: not installed")
try:
    if m.version("gtsrb-knowledge-distillation") != "2.0.0":
        problems.append("Project installation is out of date")
except m.PackageNotFoundError:
    problems.append("Project package is not installed")
if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32 or sys.prefix == sys.base_prefix:
    problems.append("A 64-bit Python 3.12 virtual environment is required")
print(json.dumps({"healthy": not problems, "problems": problems}))
sys.exit(bool(problems))
'''


def require_inside(root: Path, target: Path) -> Path:
    root = root.resolve()
    target = target.resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Path must remain inside the project: {target}")
    return target


def child_environment(root: Path) -> dict[str, str]:
    """Disable global Python/pip redirections and local GPU visibility."""
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        env.pop(key, None)
    allowed_pip = {"PIP_CERT", "PIP_CLIENT_CERT", "PIP_PROXY", "PIP_TIMEOUT", "PIP_DEFAULT_TIMEOUT", "PIP_RETRIES"}
    for key in list(env):
        if key.startswith("PIP_") and key not in allowed_pip:
            env.pop(key, None)
    temporary = require_inside(root, root / ".local" / "temp")
    temporary.mkdir(parents=True, exist_ok=True)
    require_inside(root, temporary)
    env.update(TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary),
               PYTHONNOUSERSITE="1", PYTHONUTF8="1", PIP_CONFIG_FILE=os.devnull,
               CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2", MPLBACKEND="Agg")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        env.setdefault(key, "1")
    return env


def environment_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def healthy(venv: Path, root: Path, env: dict[str, str]) -> bool:
    python = environment_python(venv)
    config = venv / "pyvenv.cfg"
    if not python.is_file() or not config.is_file():
        return False
    if "include-system-site-packages = true" in config.read_text(encoding="utf-8").lower():
        return False
    try:
        result = subprocess.run([str(python), "-I", "-c", PROBE, str(root / "requirements-runtime.txt")],
                                env=env, cwd=root, text=True, capture_output=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def setup(root: Path, venv: Path, *, repair: bool = False) -> Path:
    root = root.resolve()
    venv = require_inside(root, venv)
    if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise ValueError("Run setup with 64-bit Python 3.12; the launcher can obtain it automatically.")
    env = child_environment(root)
    if healthy(venv, root, env):
        return environment_python(venv)
    if venv.exists():
        if not repair:
            raise ValueError("Existing environment is incompatible; use --repair to preserve and replace it.")
        backups = require_inside(root, root / ".local" / "environment-backups")
        backups.mkdir(parents=True, exist_ok=True)
        backup = require_inside(root, backups / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]))
        require_inside(root, venv).rename(backup)
    subprocess.run([sys.executable, "-I", "-m", "venv", str(venv)], check=True, cwd=root, env=env)
    python = environment_python(venv)
    pip = [str(python), "-I", "-m", "pip", "--disable-pip-version-check", "--no-cache-dir"]
    subprocess.run(pip + ["install", "--index-url", "https://pypi.org/simple", "-r", str(root / "requirements-runtime.txt")],
                   check=True, cwd=root, env=env)
    subprocess.run(pip + ["install", "--no-deps", "--no-build-isolation", "-e", str(root)], check=True, cwd=root, env=env)
    subprocess.run([str(python), "-I", "-m", "pip", "check"], check=True, cwd=root, env=env)
    if not healthy(venv, root, env):
        raise RuntimeError("Installation finished but pinned runtime validation failed.")
    return python


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=PROJECT_ROOT / ".venv")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()
    try:
        python = setup(PROJECT_ROOT, args.venv, repair=args.repair)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Runtime setup failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"ready": True, "python": str(python), "device": "cpu"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
