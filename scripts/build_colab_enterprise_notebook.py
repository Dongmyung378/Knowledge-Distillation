"""Build a self-contained Google Cloud Colab Enterprise (L4) one-shot Jupyter notebook.

The generated notebook embeds a SHA-256-verified ``tar.gz`` snapshot of ``seg_kd/``,
``configs/l4-plan.json``, ``scripts/run_l4_study.py``, ``tests/``, ``pyproject.toml``,
and ``requirements-l4.txt`` so users can upload ONLY the single ``.ipynb`` file to
Vertex AI Colab Enterprise and click **Run All** without Google Drive or Git setup.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NOTEBOOK = REPO_ROOT / "notebooks" / "colab_enterprise_l4_one_shot.ipynb"

BUNDLE_GLOBS: tuple[str, ...] = (
    "pyproject.toml",
    "requirements-l4.txt",
    "configs/l4-plan.json",
    "seg_kd/*.py",
    "scripts/run_l4_study.py",
    "tests/*.py",
)


def _build_embedded_bundle(repo_root: Path) -> tuple[str, str, list[str]]:
    files: list[Path] = []
    for pattern in BUNDLE_GLOBS:
        files.extend(sorted(repo_root.glob(pattern)))
    unique_files = sorted({p.resolve() for p in files if p.is_file()}, key=lambda p: p.relative_to(repo_root).as_posix())
    if not unique_files:
        raise RuntimeError("No source files matched BUNDLE_GLOBS")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for path in unique_files:
            rel = path.relative_to(repo_root).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            info.mtime = 0  # Deterministic archive bytes across builds
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))

    raw_bytes = buf.getvalue()
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    rel_names = [p.relative_to(repo_root).as_posix() for p in unique_files]
    return b64, sha256, rel_names


def _lines(text: str) -> list[str]:
    stripped = text.strip("\n")
    if not stripped:
        return []
    parts = stripped.splitlines()
    return [line + "\n" for line in parts[:-1]] + [parts[-1]]


def _md(cell_id: str, text: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": _lines(text)}


def _code(cell_id: str, text: str, *, colab_form: bool = False) -> dict:
    meta = {"cellView": "both"} if colab_form else {}
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": meta,
        "execution_count": None,
        "outputs": [],
        "source": _lines(text),
    }


def _chunk_b64(b64: str, width: int = 100) -> str:
    chunks = [b64[i : i + width] for i in range(0, len(b64), width)]
    return "\n".join(f'    "{chunk}"' for chunk in chunks)


def build_notebook() -> Path:
    b64, bundle_sha, rel_files = _build_embedded_bundle(REPO_ROOT)
    file_list_md = ", ".join(f"`{name}`" for name in rel_files[:6]) + f" 등 총 {len(rel_files)}개 파일"

    cells = [
        _md(
            "intro",
            f"""
# 자율주행 도로 장면 Semantic Segmentation KD — Colab Enterprise (L4) 원샷 실행 노트북

이 노트북은 **Vertex AI Colab Enterprise (NVIDIA L4 24GB)**에서 **단일 `.ipynb` 파일 업로드 후 `모두 실행(Run All)` 한 번으로 전체 실험을 완주**할 수 있도록 설계된 완전 자립형 노트북입니다.

### 일반 Colab 노트북과의 차이점
- **Google Drive 마운트 미사용**: Colab Enterprise VM 로컬 디스크 + **네이티브 GCS(`gs://`)** 동기화 지원
- **코드 내장형 (Zero Git / Zero Upload)**: `seg_kd/` 전체 패키지, `configs/l4-plan.json`, `scripts/run_l4_study.py`, 단위 테스트({file_list_md})가 SHA-256 검증 번들로 내장되어 별도 코드 업로드가 필요 없습니다.
- **로그인 없는 실제 주행 데이터 자동 다운로드**:
  - 기본값 `hf_cityscapes`: HuggingFace **Cityscapes-19** (`Chris1/cityscapes` 고해상도 `384×768`, 2,975 train / 500 val) 자동 다운로드
  - 대안 `hf_bdd100k`: HuggingFace **BDD100K-Semantic-Segmentation** (19클래스) 자동 다운로드
  - 대안 `gcs`: GCP Cloud Storage(`gs://...`)의 공식 Cityscapes / BDD100K ZIP 고속 복사
  - **KITTI Semantics Zero-Shot**(`avg-kitti/data_semantics.zip`): 공식 S3에서 로그인 없이 자동 다운로드해 교차 데이터셋 일반화까지 검증

### 권장 런타임 템플릿 (Colab Enterprise)
- 머신 유형: **`g2-standard-4`** 또는 **`g2-standard-8`** + **NVIDIA L4 (24GB)**, 디스크 `100GB+`
- 상단 셀 1에서 `RUN_MODE`만 고르고 **런타임 → 모두 실행 (Run All)**을 누르면 끝까지 실행됩니다.
""",
        ),
        _md(
            "sec-config",
            """
## 1. 원샷 실행 설정 (`Run All` 전에 여기만 확인)

- **`RUN_MODE`**:
  - `pilot` *(기본 추천, 약 1.5시간)*: 단일 세션에서 `SegFormer-B5` 교사 + 5종 KD Ablation(`baseline`, `pixel_kl`, `cwd`, `cwd_feat`, `cwd_safety`, `384×768`) + Pareto + QAT-KD + ONNX FP32/INT8 + KITTI Zero-Shot + L4 BF16 FPS + 시각화 + ZIP 번들까지 완주
  - `full` *(약 4~6시간, 자동 재개)*: `configs/l4-plan.json` 전체 3시드(`378, 379, 380`), `384×768`, 35~45에포크, Pareto 3크기(`mobilenetv4_small`, `mobilenetv3_large`, `mobilenetv4_medium`)
  - `smoke` *(약 1~2분)*: 합성 도로 데이터로 전체 파이프라인(학습→QAT→ONNX FP32/INT8→동결→평가→시각화) 즉시 점검
- **`GCS_OUTPUT_URI`**: 비워두면(`""`) VM 로컬에 저장 후 최종 ZIP(`release-bundle.zip`) 생성. `gs://버킷/경로`를 입력하면 단계별 완료마다 GCS로 자동 백업 및 재개합니다.
""",
        ),
        _code(
            "cell-config",
            """
#@title Colab Enterprise L4 원샷 설정 { display-mode: "form" }
RUN_MODE = "pilot" #@param ["pilot", "full", "smoke"]
RUN_NAME = "driving-seg-kd-l4-v4" #@param {type:"string"}

DATA_SOURCE = "hf_cityscapes" #@param ["hf_cityscapes", "gcs", "kitti_direct", "hf_bdd100k", "synthetic"]
HF_BDD100K_REPO = "liqijia/BDD100K-Semantic-Segmentation" #@param {type:"string"}
GCS_DATA_URI = "" #@param {type:"string"}
GCS_DATA_FORMAT = "bdd100k" #@param ["bdd100k", "cityscapes"]

ENABLE_KITTI_ZEROSHOT = True #@param {type:"boolean"}
GCS_OUTPUT_URI = "" #@param {type:"string"}

import os
from pathlib import Path

# Colab Enterprise uses /content or /home/jupyter depending on runtime template.
if Path("/content").exists() and os.access("/content", os.W_OK):
    BASE_DIR = Path("/content/driving_seg_kd")
else:
    BASE_DIR = Path.cwd() / "driving_seg_kd"

REPO_DIR = BASE_DIR / "repo"
DATA_ROOT = BASE_DIR / "data"
OUTPUT_ROOT = BASE_DIR / "runs" / f"{RUN_NAME}-{RUN_MODE}"
MANIFEST_DIR = BASE_DIR / "manifests"

for d in (REPO_DIR, DATA_ROOT, OUTPUT_ROOT, MANIFEST_DIR):
    d.mkdir(parents=True, exist_ok=True)

EFFECTIVE_SOURCE = "synthetic" if RUN_MODE == "smoke" else DATA_SOURCE
DATASET_DIR = DATA_ROOT / EFFECTIVE_SOURCE
MANIFEST_PATH = MANIFEST_DIR / f"{EFFECTIVE_SOURCE}-manifest.json"
KITTI_DIR = DATA_ROOT / "kitti"
KITTI_MANIFEST_PATH = MANIFEST_DIR / "kitti-manifest.json"

print(f"BASE_DIR         : {BASE_DIR}")
print(f"RUN_MODE         : {RUN_MODE}")
print(f"EFFECTIVE_SOURCE : {EFFECTIVE_SOURCE}")
print(f"OUTPUT_ROOT      : {OUTPUT_ROOT}")
print(f"GCS_OUTPUT_URI   : {GCS_OUTPUT_URI or '(disabled - local VM disk only)'}")
print(f"KITTI_ZEROSHOT   : {ENABLE_KITTI_ZEROSHOT and RUN_MODE != 'smoke'}")
""",
            colab_form=True,
        ),
        _md(
            "sec-unpack",
            """
## 2. 내장 `seg_kd` 프로젝트 자동 추출 및 SHA-256 무결성 검증

외부 Git/Drive 없이 이 노트북에 내장된 전체 `seg_kd` 소스를 추출하고 체크섬을 검증합니다.
""",
        ),
        _code(
            "cell-unpack-code",
            f"""
import base64
import hashlib
import inspect
import io
import os
import sys
import tarfile

EMBEDDED_BUNDLE_SHA256 = "{bundle_sha}"
EMBEDDED_BUNDLE_B64 = (
{_chunk_b64(b64)}
)

raw_tar_gz = base64.b64decode(EMBEDDED_BUNDLE_B64.encode("ascii"))
actual_sha = hashlib.sha256(raw_tar_gz).hexdigest()
if actual_sha != EMBEDDED_BUNDLE_SHA256:
    raise RuntimeError(f"Embedded bundle checksum mismatch: {{actual_sha}} != {{EMBEDDED_BUNDLE_SHA256}}")

with tarfile.open(fileobj=io.BytesIO(raw_tar_gz), mode="r:gz") as tar:
    if "filter" in inspect.signature(tar.extractall).parameters:
        tar.extractall(path=REPO_DIR, filter="data")
    else:
        tar.extractall(path=REPO_DIR)

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

print(f"Verified & extracted {len(rel_files)} project files into {{REPO_DIR}}")
print(f"Bundle SHA-256: {{actual_sha}}")
""",
        ),
        _md(
            "sec-env",
            """
## 3. NVIDIA L4 GPU 점검, 필수 패키지 설치 및 단위 테스트 검증

Colab Enterprise 런타임에 이미 설치된 CUDA PyTorch는 유지하고, 누락된 패키지(`timm`, `transformers`, `onnx`, `onnxruntime`, `huggingface_hub`, `datasets`)만 빠르게 설치합니다.
""",
        ),
        _code(
            "cell-env-install",
            """
import importlib.util
import subprocess
import sys

print("=== [1/4] GPU Check ===")
try:
    print(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        text=True,
    ).strip())
except Exception as exc:
    print("nvidia-smi unavailable:", exc)

print("\\n=== [2/4] Installing missing dependencies ===")
if importlib.util.find_spec("torch") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"],
        check=True,
    )

needed = ["timm>=1.0.9", "transformers>=4.44.0", "onnx>=1.16.0", "onnxruntime>=1.18.0", "huggingface_hub>=0.24.0", "datasets>=2.20.0", "matplotlib>=3.8", "pillow>=10.0"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR), "--no-deps"], check=True)

print("\\n=== [3/4] seg_kd doctor ===")
subprocess.run([sys.executable, "-m", "seg_kd", "doctor"], check=True)

print("\\n=== [4/4] Unit test suite ===")
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)

# Optional: restore prior completed runs from GCS for automatic resumability
if GCS_OUTPUT_URI.strip():
    print(f"\\n=== Syncing existing checkpoints from {GCS_OUTPUT_URI} ===")
    subprocess.run(["gcloud", "storage", "rsync", "-r", GCS_OUTPUT_URI.rstrip("/"), str(OUTPUT_ROOT)], check=False)
""",
        ),
        _md(
            "sec-data",
            """
## 4. 실제 주행 데이터셋 자동 다운로드 및 SHA-256 매니페스트 생성

- `hf_bdd100k`: HuggingFace에서 로그인 없이 BDD100K 10K Semantic Segmentation 데이터를 내려받습니다(폴더 구조/Parquet 모두 자동 대응).
- `hf_cityscapes`: HuggingFace `Chris1/cityscapes` (고해상도 원본, 19클래스 Cityscapes trainId)을 PNG 이미지/마스크로 자동 전개합니다.
- `gcs`: `GCS_DATA_URI`에서 ZIP 파일을 고속 복사해 압축 해제합니다.
- `ENABLE_KITTI_ZEROSHOT=True`이면 공식 KITTI Semantics (`data_semantics.zip`, ~300MB)도 자동 다운로드합니다.
""",
        ),
        _code(
            "cell-prepare-data",
            """
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image


def _export_hf_dataset_to_bdd_layout(repo_id: str, dst_dir: Path) -> None:
    \"\"\"Convert a HuggingFace parquet/imagefolder driving dataset into standard BDD100K folders.\"\"\"
    from datasets import load_dataset

    print(f"Loading HF dataset via `datasets`: {repo_id} ...")
    ds = load_dataset(repo_id)
    split_map = {}
    for name in ds.keys():
        low = name.lower()
        if "train" in low:
            split_map["train"] = name
        elif "val" in low or "test" in low:
            split_map.setdefault("val", name)
    if "train" not in split_map:
        raise RuntimeError(f"Could not find train split in {repo_id}: {list(ds.keys())}")
    if "val" not in split_map:
        split_obj = ds[split_map["train"]].train_test_split(test_size=0.125, seed=378)
        ds = {"train": split_obj["train"], "val": split_obj["test"]}
        split_map = {"train": "train", "val": "val"}

    sample = ds[split_map["train"]][0]
    img_key = next((k for k in ("image", "img", "pixel_values", "rgb") if k in sample), None)
    mask_key = next(
        (k for k in ("segmentation_19", "label", "mask", "annotation", "semantic_mask", "labels", "segmentation") if k in sample),
        None,
    )
    if img_key is None or mask_key is None:
        raise RuntimeError(f"Could not identify image/mask columns in {repo_id}: {list(sample.keys())}")

    print(f"Exporting columns image={img_key!r}, mask={mask_key!r} -> {dst_dir}")
    for out_split, hf_split in split_map.items():
        img_dir = dst_dir / "images" / "10k" / out_split
        mask_dir = dst_dir / "labels" / "sem_seg" / "masks" / out_split
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for idx, row in enumerate(ds[hf_split]):
            stem = f"{out_split}_{idx:05d}"
            img_val = row[img_key]
            if isinstance(img_val, Image.Image):
                img_pil = img_val.convert("RGB")
            else:
                arr = np.asarray(img_val)
                if arr.ndim == 3 and arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = arr.astype(np.float32)
                    if arr.min() < 0.0:
                        denom = max(float(arr.max() - arr.min()), 1e-6)
                        arr = (arr - arr.min()) / denom * 255.0
                    elif arr.max() <= 1.05:
                        arr = arr * 255.0
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(arr, mode="RGB")

            mask_val = row[mask_key]
            if isinstance(mask_val, Image.Image):
                mask_arr = np.asarray(mask_val)
            else:
                mask_arr = np.asarray(mask_val)
            mask_arr = np.squeeze(mask_arr)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0] if mask_arr.shape[0] <= 4 else mask_arr[..., 0]
            mask_arr = mask_arr.astype(np.int64)
            mask_arr = np.where((mask_arr < 0) | (mask_arr >= 19), 255, mask_arr).astype(np.uint8)
            img_pil.save(img_dir / f"{stem}.jpg", quality=95)
            Image.fromarray(mask_arr, mode="L").save(mask_dir / f"{stem}_train_id.png")


def _download_kitti_semantics(dst_dir: Path) -> None:
    if (dst_dir / "training" / "image_2").is_dir() and (dst_dir / "training" / "semantic").is_dir():
        print(f"KITTI Semantics already ready: {dst_dir}")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dst_dir / "data_semantics.zip"
    url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip"
    print(f"Downloading official KITTI Semantics from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)
    zip_path.unlink(missing_ok=True)
    print(f"KITTI Semantics extracted to {dst_dir}")


if EFFECTIVE_SOURCE == "synthetic":
    subprocess.run(
        [
            sys.executable, "-m", "seg_kd", "prepare-data",
            "--dataset", "synthetic",
            "--data", str(DATASET_DIR),
            "--output", str(MANIFEST_PATH),
            "--image-size", "128x256",
            "--synthetic-train", "12",
            "--synthetic-val", "4",
            "--synthetic-eval", "4",
            "--seed", "378",
        ],
        check=True,
    )
elif EFFECTIVE_SOURCE == "hf_bdd100k":
    if not MANIFEST_PATH.is_file():
        from huggingface_hub import snapshot_download
        from seg_kd.data import build_bdd100k_manifest, save_manifest

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {HF_BDD100K_REPO} from HuggingFace Hub ...")
        snapshot_download(
            repo_id=HF_BDD100K_REPO,
            repo_type="dataset",
            local_dir=str(DATASET_DIR),
        )
        # Extract any bundled zip archives inside the HF snapshot
        for zp in sorted(DATASET_DIR.rglob("*.zip")):
            marker = zp.with_suffix(".extracted")
            if not marker.exists():
                print(f"Extracting {zp.name} ...")
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(zp.parent)
                marker.write_text("ok", encoding="utf-8")
        try:
            manifest = build_bdd100k_manifest(DATASET_DIR, seed=378, val_ratio=0.15, hash_files=False)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Raw folder layout not found ({exc}); falling back to HuggingFace `datasets` export ...")
            export_dir = DATA_ROOT / "hf_bdd100k_exported"
            _export_hf_dataset_to_bdd_layout(HF_BDD100K_REPO, export_dir)
            DATASET_DIR = export_dir
            manifest = build_bdd100k_manifest(DATASET_DIR, seed=378, val_ratio=0.15, hash_files=False)
        save_manifest(manifest, MANIFEST_PATH)
        print("Built BDD100K manifest:", manifest["counts"])
    else:
        if (DATA_ROOT / "hf_bdd100k_exported").is_dir():
            DATASET_DIR = DATA_ROOT / "hf_bdd100k_exported"
        print(f"Reusing existing manifest: {MANIFEST_PATH} (data={DATASET_DIR})")
elif EFFECTIVE_SOURCE == "hf_cityscapes":
    if not MANIFEST_PATH.is_file():
        from seg_kd.data import build_bdd100k_manifest, save_manifest

        _export_hf_dataset_to_bdd_layout("Chris1/cityscapes", DATASET_DIR)
        manifest = build_bdd100k_manifest(DATASET_DIR, seed=378, val_ratio=0.15, hash_files=False)
        save_manifest(manifest, MANIFEST_PATH)
        print("Built HF Cityscapes-19 manifest:", manifest["counts"])
    else:
        print(f"Reusing existing manifest: {MANIFEST_PATH} (data={DATASET_DIR})")
elif EFFECTIVE_SOURCE == "gcs":
    if not GCS_DATA_URI.strip():
        raise ValueError("DATA_SOURCE='gcs' 선택 시 GCS_DATA_URI(예: gs://bucket/bdd100k)를 입력해야 합니다.")
    if not DATASET_DIR.is_dir() or not any(DATASET_DIR.iterdir()):
        zip_stage = DATA_ROOT / "_gcs_zips"
        zip_stage.mkdir(parents=True, exist_ok=True)
        subprocess.run(["gcloud", "storage", "cp", "-r", GCS_DATA_URI.rstrip("/") + "/*", str(zip_stage)], check=True)
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        zips = sorted(zip_stage.rglob("*.zip"))
        if zips:
            for zp in zips:
                print(f"Extracting {zp.name} -> {DATASET_DIR}")
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(DATASET_DIR)
            shutil.rmtree(zip_stage, ignore_errors=True)
        else:
            shutil.copytree(zip_stage, DATASET_DIR, dirs_exist_ok=True)
    if not MANIFEST_PATH.is_file():
        subprocess.run(
            [
                sys.executable, "-m", "seg_kd", "prepare-data",
                "--dataset", GCS_DATA_FORMAT,
                "--data", str(DATASET_DIR),
                "--output", str(MANIFEST_PATH),
                "--seed", "378",
                "--no-hash",
            ],
            check=True,
        )
elif EFFECTIVE_SOURCE == "kitti_direct":
    from seg_kd.data import build_bdd100k_manifest, save_manifest
    from seg_kd.taxonomy import map_mask_to_train_ids

    _download_kitti_semantics(KITTI_DIR)
    if not MANIFEST_PATH.is_file():
        # Partition KITTI 200 images into official train (160) + locked_eval (40) via BDD layout
        imgs = sorted((KITTI_DIR / "training" / "image_2").glob("*.png"))
        for idx, img_p in enumerate(imgs):
            mask_p = KITTI_DIR / "training" / "semantic" / img_p.name
            out_split = "val" if idx % 5 == 0 else "train"
            (DATASET_DIR / "images" / "10k" / out_split).mkdir(parents=True, exist_ok=True)
            (DATASET_DIR / "labels" / "sem_seg" / "masks" / out_split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_p, DATASET_DIR / "images" / "10k" / out_split / img_p.name)
            with Image.open(mask_p) as raw_m:
                tid = map_mask_to_train_ids(np.asarray(raw_m), label_format="label_id")
            Image.fromarray(tid, mode="L").save(
                DATASET_DIR / "labels" / "sem_seg" / "masks" / out_split / f"{img_p.stem}_train_id.png"
            )
        manifest = build_bdd100k_manifest(DATASET_DIR, seed=378, val_ratio=0.18, hash_files=True)
        save_manifest(manifest, MANIFEST_PATH)
        print("Built KITTI direct manifest:", manifest["counts"])

# Always stage KITTI Semantics for cross-dataset zero-shot evaluation unless disabled/smoke/kitti_direct
USE_KITTI_ZEROSHOT = ENABLE_KITTI_ZEROSHOT and RUN_MODE != "smoke" and EFFECTIVE_SOURCE != "kitti_direct"
if USE_KITTI_ZEROSHOT:
    try:
        _download_kitti_semantics(KITTI_DIR)
        if not KITTI_MANIFEST_PATH.is_file():
            subprocess.run(
                [
                    sys.executable, "-m", "seg_kd", "prepare-data",
                    "--dataset", "kitti",
                    "--data", str(KITTI_DIR),
                    "--output", str(KITTI_MANIFEST_PATH),
                ],
                check=True,
            )
    except Exception as exc:
        print(f"WARNING: KITTI zero-shot download failed ({exc}); continuing without KITTI zero-shot.")
        USE_KITTI_ZEROSHOT = False
""",
        ),
        _md(
            "sec-run",
            """
## 5. L4 전체 파이프라인 원샷 실행 (`teacher → ablation → pareto → export → freeze → eval → plots`)

- `RUN_MODE`에 맞춰 계획을 고정 저장한 뒤 단계별로 순차 실행합니다.
- 각 단계가 끝날 때 `GCS_OUTPUT_URI`가 설정되어 있으면 자동으로 GCS에 백업합니다.
- 이미 완료된 런은 자동 건너뛰므로 중단 후 다시 `Run All`을 눌러도 처음부터 다시 돌지 않습니다.
""",
        ),
        _code(
            "cell-run-all",
            """
import json
import subprocess
import sys

plan = json.loads((REPO_DIR / "configs" / "l4-plan.json").read_text(encoding="utf-8"))

if RUN_MODE == "pilot":
    plan["protocol"] = "driving-seg-kd-l4-v4-colab-enterprise-pilot"
    plan["seeds"] = [378]
    plan["deployment_seed"] = 378
    plan["image_size"] = [384, 768]
    plan["teacher"]["epochs"] = 10
    plan["teacher"]["batch_size"] = 8
    plan["main_student"]["epochs"] = 35
    plan["main_student"]["batch_size"] = 16
    plan["main_student"]["qat_epochs"] = 6
    plan["ablation_methods"] = ["baseline", "pixel_kl", "cwd", "cwd_feat"]
    plan["pareto_students"] = [
        {"arch": "mobilenetv4_small", "fpn_dim": 96},
        {"arch": "mobilenetv3_large", "fpn_dim": 128},
        {"arch": "mobilenetv4_medium", "fpn_dim": 128},
    ]
elif RUN_MODE == "full":
    plan["image_size"] = [384, 768]

ACTIVE_PLAN_PATH = OUTPUT_ROOT / "active-plan.json"
ACTIVE_PLAN_PATH.write_text(json.dumps(plan, indent=2) + "\\n", encoding="utf-8")
print("Active Plan:", json.dumps({
    "protocol": plan["protocol"],
    "image_size": plan["image_size"],
    "seeds": plan["seeds"],
    "teacher_epochs": plan["teacher"]["epochs"],
    "student_epochs": plan["main_student"]["epochs"],
    "ablation_methods": plan["ablation_methods"],
    "qat_epochs": plan["main_student"]["qat_epochs"],
}, indent=2))


def _sync_gcs() -> None:
    if GCS_OUTPUT_URI.strip():
        print(f"Syncing outputs -> {GCS_OUTPUT_URI} ...")
        subprocess.run(["gcloud", "storage", "rsync", "-r", str(OUTPUT_ROOT), GCS_OUTPUT_URI.rstrip("/")], check=False)


stage_groups = [
    ["teacher"],
    ["ablation"],
    ["pareto"],
    ["export"],
    ["freeze", "eval", "plots"],
]

for group in stage_groups:
    print(f"\\n====================================================")
    print(f"Running stages: {group}")
    print(f"====================================================", flush=True)
    cmd = [
        sys.executable, "scripts/run_l4_study.py",
        "--plan", str(ACTIVE_PLAN_PATH),
        "--manifest", str(MANIFEST_PATH),
        "--data", str(DATASET_DIR),
        "--output", str(OUTPUT_ROOT),
        "--stage", *group,
        "--device", "auto",
    ]
    if RUN_MODE == "smoke":
        cmd.append("--smoke")
    if USE_KITTI_ZEROSHOT and "eval" in group and KITTI_MANIFEST_PATH.is_file():
        cmd.extend(["--kitti-manifest", str(KITTI_MANIFEST_PATH), "--kitti-data", str(KITTI_DIR)])
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    _sync_gcs()

print("\\nALL STAGES COMPLETED SUCCESSFULLY!")
""",
        ),
        _md(
            "sec-summary",
            """
## 6. 최종 결과 요약표 & L4 CUDA BF16 실시간 FPS 벤치마크
""",
        ),
        _code(
            "cell-summary-and-gpu-bench",
            """
import json
import torch
from seg_kd.benchmark import benchmark_model
from seg_kd.training import write_json

sel = json.loads((OUTPUT_ROOT / "selection-report.json").read_text(encoding="utf-8"))
summary = json.loads((OUTPUT_ROOT / "release-summary.json").read_text(encoding="utf-8"))

print("=== [1] Validation Selection Report ===")
print(f"Baseline val mIoU : {sel['baseline_mean_val_miou']*100:.2f}%")
print(f"Selected Method   : {sel['selected_method']}")
for c in sel["candidates"]:
    print(
        f"  {c['method']:12s} | val_mIoU={c['mean_val_miou']*100:6.2f}% (±{c['std_val_miou']*100:.2f}%p) "
        f"| gain={c['mean_gain_over_baseline_pp']:+6.2f}%p | safety={c['mean_val_safety_miou']*100:6.2f}% | ECE={c['mean_val_ece']:.4f}"
    )

print("\\n=== [2] Locked Eval + CPU ONNX/PyTorch Benchmarks ===")
print(f"{'Model':24s} | {'mIoU':>7s} | {'Safety':>7s} | {'PixAcc':>7s} | {'ECE':>6s} | {'CPU p50':>9s} | {'CPU FPS':>7s} | {'MiB':>6s}")
print("-" * 92)
for label, m in summary["locked_eval"].items():
    b = summary["benchmarks_cpu"].get(label, {})
    p50 = b.get("latency_ms", {}).get("p50", float("nan"))
    fps = b.get("latency_ms", {}).get("fps_p50", float("nan"))
    mib = b.get("model_mib", float("nan"))
    print(
        f"{label:24s} | {m['miou']*100:6.2f}% | {m['safety_critical_miou']*100:6.2f}% | "
        f"{m['pixel_accuracy']*100:6.2f}% | {m['ece']:6.4f} | {p50:7.2f}ms | {fps:7.2f} | {mib:6.2f}"
    )

for gate_path in sorted((OUTPUT_ROOT / "export").rglob("int8-acceptance.json")):
    print(f"\\n=== [3] Validation-Only INT8 Acceptance Gate ({gate_path.parent.name}) ===")
    print(json.dumps(json.loads(gate_path.read_text(encoding="utf-8")), indent=2))

if summary.get("zero_shot_kitti"):
    print("\\n=== [4] KITTI Semantics Zero-Shot Cross-Dataset Evaluation ===")
    for label, km in summary["zero_shot_kitti"].items():
        print(f"  {label:24s} | mIoU={km['miou']*100:6.2f}% | Safety={km['safety_critical_miou']*100:6.2f}% | ECE={km['ece']:.4f}")

if torch.cuda.is_available():
    plan = json.loads(ACTIVE_PLAN_PATH.read_text(encoding="utf-8"))
    img_hw = (128, 256) if RUN_MODE == "smoke" else tuple(plan["image_size"])
    gpu_bench = {}
    print(f"\\n=== [5] L4 CUDA BF16 Real-Time FPS ({img_hw[0]}x{img_hw[1]}) ===")
    for ckpt in sorted(OUTPUT_ROOT.rglob("*.pt")):
        if ckpt.name not in {"teacher.pt", "student.pt", "student-qat.pt"}:
            continue
        rel = str(ckpt.relative_to(OUTPUT_ROOT))
        res = benchmark_model(
            ckpt,
            image_size=img_hw,
            batch_size=1,
            warmup=3 if RUN_MODE == "smoke" else 15,
            runs=10 if RUN_MODE == "smoke" else 60,
            device_spec="cuda",
            precision="bf16",
        )
        gpu_bench[rel] = res
        print(
            f"  {rel:48s} | p50={res['latency_ms']['p50']:6.2f}ms | "
            f"{res['latency_ms']['fps_p50']:7.1f} FPS | params={res['parameter_count']:,}"
        )
    write_json(gpu_bench, OUTPUT_ROOT / "benchmarks" / "l4-cuda-bf16.json")
""",
        ),
        _md(
            "sec-plots",
            """
## 7. 도로 장면 오버레이·오차 맵·Pareto 곡선·안전 클래스 IoU 인라인 확인
""",
        ),
        _code(
            "cell-plots",
            """
import json
from IPython.display import Image, display

summary = json.loads((OUTPUT_ROOT / "release-summary.json").read_text(encoding="utf-8"))
sel = json.loads((OUTPUT_ROOT / "selection-report.json").read_text(encoding="utf-8"))
best_label = f"student_{sel['selected_method']}"

if "student_baseline" in summary["locked_eval"] and best_label in summary["locked_eval"]:
    b_slices = summary["locked_eval"]["student_baseline"].get("slices", {})
    d_slices = summary["locked_eval"][best_label].get("slices", {})
    common = sorted(set(b_slices) & set(d_slices))
    if common:
        print(f"=== Weather / Time-of-Day Slices (Baseline vs {best_label}) ===")
        for k in common:
            bm = b_slices[k]["miou"] * 100
            dm = d_slices[k]["miou"] * 100
            print(f"  {k:24s} | Baseline={bm:6.2f}% | Distilled={dm:6.2f}% | Delta={dm - bm:+6.2f}%p")

fig_dir = OUTPUT_ROOT / "figures"
for name in ("road-scene-overlays.png", "ablation-and-pareto.png", "per-class-safety-gain.png"):
    p = fig_dir / name
    if p.is_file():
        print(f"\\n--- {name} ---")
        display(Image(filename=str(p)))
""",
        ),
        _md(
            "sec-bundle",
            """
## 8. 최종 결과 번들(`release-bundle.zip`) 패키징 및 GCS 업로드

완료된 ZIP 파일은 Colab Enterprise 왼쪽 파일 탐색기에서 우클릭 → **다운로드** 하거나, `GCS_OUTPUT_URI`를 지정한 경우 GCS 버킷에서 바로 내려받을 수 있습니다.
""",
        ),
        _code(
            "cell-bundle",
            """
import subprocess
import zipfile

bundle_path = BASE_DIR / f"{OUTPUT_ROOT.name}-release-bundle.zip"
include_patterns = [
    "active-plan.json",
    "selection-report.json",
    "evaluation-freeze.json",
    "release-summary.json",
    "**/training-report.json",
    "**/history.json",
    "evaluations/**/*.json",
    "benchmarks/**/*.json",
    "figures/*.png",
    "export/**/*.json",
    "export/**/*.onnx",
]

added = []
with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for pattern in include_patterns:
        for file_path in sorted(OUTPUT_ROOT.glob(pattern)):
            if file_path.is_file():
                arcname = file_path.relative_to(OUTPUT_ROOT).as_posix()
                if arcname not in added:
                    zf.write(file_path, arcname=arcname)
                    added.append(arcname)

print(f"Release bundle ready: {bundle_path} ({bundle_path.stat().st_size / (1024**2):.2f} MiB, {len(added)} files)")
for item in added[:30]:
    print("  -", item)
if len(added) > 30:
    print(f"  ... (+{len(added) - 30} more)")

if GCS_OUTPUT_URI.strip():
    dest_uri = GCS_OUTPUT_URI.rstrip("/") + "/" + bundle_path.name
    subprocess.run(["gcloud", "storage", "cp", str(bundle_path), dest_uri], check=True)
    print(f"Uploaded bundle to GCS: {dest_uri}")
""",
        ),
    ]

    # Validate AST syntax of every code cell!
    for cell in cells:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell:{cell['id']}")

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": OUTPUT_NOTEBOOK.name,
                "provenance": [],
                "gpuType": "L4",
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "cells": cells,
    }

    OUTPUT_NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"Built {OUTPUT_NOTEBOOK} ({OUTPUT_NOTEBOOK.stat().st_size / 1024:.1f} KiB, "
        f"{len(cells)} cells, {len(rel_files)} embedded files, bundle_sha256={bundle_sha[:16]}...)"
    )
    return OUTPUT_NOTEBOOK


if __name__ == "__main__":
    build_notebook()
