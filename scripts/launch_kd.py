"""Install into the project, then invoke the CPU CLI without global changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.setup_runtime import child_environment, setup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--runtime-venv", type=Path, default=PROJECT_ROOT / ".venv")
    args, application_args = parser.parse_known_args()
    if application_args[:1] == ["train"]:
        print("Full training runs on Kaggle. Use its GPU-enabled kernel with python -m knowledge_distillation train ...", file=sys.stderr)
        return 1
    if not application_args:
        application_args = ["doctor"]
    try:
        python = setup(PROJECT_ROOT, args.runtime_venv, repair=True)
        if args.setup_only:
            print(json.dumps({"ready": True, "python": str(python), "device": "cpu"}))
            return 0
        return subprocess.run([str(python), "-I", "-m", "knowledge_distillation", *application_args],
                              env=child_environment(PROJECT_ROOT), cwd=PROJECT_ROOT).returncode
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Knowledge distillation could not start: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
