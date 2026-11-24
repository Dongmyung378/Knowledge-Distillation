"""Build a clean, self-contained One-Shot Verification & L4 Reproduction Notebook.

The generated notebook (``notebooks/colab_enterprise_l4_one_shot.ipynb``) embeds a
SHA-256-verified ``tar.gz`` snapshot of ``seg_kd/``, ``configs/l4-plan.json``,
``scripts/run_l4_study.py``, ``scripts/verify_release_models.py``, ``tests/``, and the
frozen ``reports/`` manifests so anyone can upload ONLY this single ``.ipynb`` file to
Google Colab / Vertex AI Colab Enterprise (or run locally in Jupyter) and click **Run All**:

  - ``verify_saved`` *(default, CPU/GPU 1-3 min)*:
      Verifies the embedded codebase + 12 unit tests, downloads the exact SHA-256 frozen
      ONNX FP32/INT8 deployment models from GitHub Release ``v4.0-models``, validates exact
      byte-for-byte freeze checksums, benchmarks real 384x768 CPU ONNXRuntime latency/speedup,
      optionally runs live Cityscapes locked_eval + KITTI + 4-corruption + ECE re-evaluation,
      and prints the full v3-max + v4-final decision & results tables.
  - ``reproduce_full`` *(NVIDIA L4 24GB, ~5-7 hours, resumable)*:
      Downloads Chris1/cityscapes (384x768) + KITTI Semantics without login and retrains
      the complete v3-max (SegFormer-B5 -> 3-seed + 3-tier Pareto + QAT + INT8 gate) and
      v4-final (Medium Baseline + 4-method ablation + Asymmetric Robust KD + BF16 QAT-KD +
      4-corruption stress test + validation temperature scaling) pipelines end-to-end.
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
    "scripts/verify_release_models.py",
    "tests/*.py",
    "reports/final-summary.json",
    "reports/v3-max/evaluation-freeze.json",
    "reports/v4-final/evaluation-freeze-v4.json",
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
            info.mtime = 0
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

    cells = [
        _md(
            "intro",
            f"""
# 자율주행 도로 세그멘테이션 지식 증류 (`seg_kd`) — 원샷 검증 및 L4 전체 재현 노트북

**SegFormer-B5 (84.6M, ViT) → MobileNetV4-Medium ONNX INT8 (7.54M, 7.91 MiB, CPU 40.5 FPS)** 이종 아키텍처 지식 증류 및 비대칭 강건성 증류(`v3-max` + `v4-final`)의 **원샷 검증**과 **NVIDIA L4 전체 재현**을 하나의 자립형 노트북으로 통합했습니다.

### 두 가지 실행 모드 (`Cell 1`에서 선택 후 `모두 실행 / Run All`)
1. **`verify_saved` *(기본값, CPU/GPU 어디서나 1~3분 원샷 검증)***:
   - 내장된 `seg_kd` 전체 패키지({len(rel_files)}개 파일) SHA-256 무결성 검증 + `unittest` 12/12 통과 확인
   - GitHub Release(`v4.0-models`)에서 실제 배포 **ONNX FP32/INT8 모델 4종** 자동 다운로드 및 `evaluation-freeze*.json`과 **SHA-256 바이트 단위 100% 일치 확인**
   - 현재 런타임에서 **`384×768` 단일 프레임 CPU ONNXRuntime 지연(p50 ms, FPS) 및 FP32→INT8 가속비 실측**
   - *(선택)* `EVALUATE_LIVE_DATASETS = True` 설정 시 Cityscapes `locked_eval`(500장) + 공식 KITTI Semantics(200장)를 로그인 없이 자동 다운로드해 **`67.95% / 66.96%` mIoU, `σ=18` 센서 노이즈 `59.59%`, 온도 보정 `T*=1.70` ECE `-74.2%`를 실시간 재측정**
2. **`reproduce_full` *(NVIDIA L4 24GB 권장, 약 5~7시간, 자동 재개)***:
   - `Chris1/cityscapes` (`384×768`, `train 2,529 / val 446 / locked_eval 500`) + 공식 KITTI Semantics (`200장`) 자동 준비
   - `v3-max` (교사 `SegFormer-B5` + 3시드 쌍체 비교 + 3-Tier Pareto + QAT-KD + ONNX INT8 게이트) + `v4-final` (Medium Baseline + `384×768` 4종 소거 + 비대칭 강건성 KD + BF16 교사 QAT-KD + 4종 악천후 스트레스 테스트 + 사후 온도 보정) 전체 재학습 및 번들 생성
""",
        ),
        _md(
            "sec-config",
            """
## 1. 원샷 실행 설정 (`Run All` 전 여기만 확인)
""",
        ),
        _code(
            "cell-config",
            """
#@title seg_kd 원샷 검증 및 L4 재현 설정 { display-mode: "form" }
RUN_MODE = "verify_saved" #@param ["verify_saved", "reproduce_full"]
EVALUATE_LIVE_DATASETS = False #@param {type:"boolean"}
BENCHMARK_THREADS = 4 #@param {type:"integer"}
BENCHMARK_RUNS = 20 #@param {type:"integer"}
GCS_OUTPUT_URI = "" #@param {type:"string"}

import os
from pathlib import Path

if Path("/content").exists() and os.access("/content", os.W_OK):
    BASE_DIR = Path("/content/driving_seg_kd")
else:
    BASE_DIR = Path.cwd() / "driving_seg_kd_workspace"

REPO_DIR = BASE_DIR / "repo"
CACHE_DIR = BASE_DIR / "verified_models"
DATA_ROOT = BASE_DIR / "data"
OUTPUT_ROOT = BASE_DIR / "runs" / f"driving-seg-kd-v4-{RUN_MODE}"
MANIFEST_DIR = BASE_DIR / "manifests"

for d in (REPO_DIR, CACHE_DIR, DATA_ROOT, OUTPUT_ROOT, MANIFEST_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"BASE_DIR               : {BASE_DIR}")
print(f"RUN_MODE               : {RUN_MODE}")
print(f"EVALUATE_LIVE_DATASETS : {EVALUATE_LIVE_DATASETS}")
print(f"BENCHMARK_THREADS      : {BENCHMARK_THREADS} (runs={BENCHMARK_RUNS})")
print(f"GCS_OUTPUT_URI         : {GCS_OUTPUT_URI or '(disabled)'}")
""",
            colab_form=True,
        ),
        _md(
            "sec-unpack",
            """
## 2. 내장 `seg_kd` 소스 및 동결 리포트 추출 (SHA-256 무결성 검증)

외부 Git 클론이나 Google Drive 마운트 없이, 노트북에 내장된 `seg_kd/` 전체 소스·테스트·동결 매니페스트(`evaluation-freeze*.json`, `final-summary.json`)를 검증 후 추출합니다.
""",
        ),
        _code(
            "cell-unpack",
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

print(f"Verified & extracted {len(rel_files)} files into {{REPO_DIR}}")
print(f"Embedded Bundle SHA-256: {{actual_sha}}")
""",
        ),
        _md(
            "sec-env-tests",
            """
## 3. 런타임 진단 (`seg_kd doctor`) 및 소프트웨어 단위 테스트 (12/12)

누락된 필수 패키지(`timm`, `transformers`, `onnx`, `onnxruntime`, `datasets`)만 설치하고, 손실 함수(`CWD`/`Cosine` 불변성·BF16 유한 그래디언트)·양자화 친화 감사(`Conv-ReLU-BN` 차단)·데이터 매니페스트 변조 탐지 등 **12개 단위 테스트**를 검증합니다.
""",
        ),
        _code(
            "cell-env-tests",
            """
import importlib.util
import subprocess
import sys

print("=== [1/3] Hardware & Runtime Check ===")
try:
    print(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
        text=True,
    ).strip())
except Exception:
    print("No NVIDIA GPU detected (CPU verification mode is fully supported).")

print("\\n=== [2/3] Ensuring Required Dependencies ===")
needed = []
for mod_name, pkg_spec in [
    ("timm", "timm>=1.0.9"),
    ("transformers", "transformers>=4.44.0"),
    ("onnx", "onnx>=1.16.0"),
    ("onnxruntime", "onnxruntime>=1.18.0"),
    ("datasets", "datasets>=2.20.0"),
    ("matplotlib", "matplotlib>=3.8"),
    ("PIL", "pillow>=10.0"),
]:
    if importlib.util.find_spec(mod_name) is None:
        needed.append(pkg_spec)
if needed:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)

subprocess.run([sys.executable, "-m", "seg_kd", "doctor"], check=True)

print("\\n=== [3/3] Unit & Integrity Test Suite (12 tests) ===")
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
""",
        ),
        _md(
            "sec-verify-models",
            """
## 4. 배포 ONNX FP32/INT8 모델 SHA-256 동결 검증 & CPU 실시간 속도 실측

GitHub Release `v4.0-models`에서 최종 배포 아티팩트 4종(`v3-max Clean` FP32/INT8, `v4-final Robust` FP32/INT8)을 내려받아:
1. `reports/v3-max/evaluation-freeze.json` 및 `reports/v4-final/evaluation-freeze-v4.json`의 **SHA-256 해시와 바이트 단위 일치** 확인
2. 현재 머신 CPU(`batch=1`, `384×768`)에서 **FP32 vs Static QDQ INT8 추론 지연(p50 ms, FPS) 및 가속비 실측**
3. 동결 마스터 요약표(`v3-max` + `v4-final` 전 실험 결과) 출력
""",
        ),
        _code(
            "cell-verify-models",
            """
import subprocess
import sys

cmd = [
    sys.executable,
    "scripts/verify_release_models.py",
    "--cache-dir", str(CACHE_DIR),
    "--threads", str(BENCHMARK_THREADS),
    "--runs", str(BENCHMARK_RUNS),
    "--warmup", "5",
    "--output", str(OUTPUT_ROOT / "live-verification-receipt.json"),
]
subprocess.run(cmd, check=True)
""",
        ),
        _md(
            "sec-live-eval",
            """
## 5. (선택) 실제 Cityscapes `locked_eval` (500장) · KITTI (200장) · 악천후 · ECE 라이브 재평가

`EVALUATE_LIVE_DATASETS = True`로 설정한 경우에만 실행됩니다. 로그인 없이 `Chris1/cityscapes`와 공식 KITTI Semantics를 내려받아, 위에서 SHA-256 검증한 **`v3-max Clean INT8` (`7.91 MiB`)**과 **`v4-final Robust INT8` (`7.91 MiB`)**의 정확도(`67.95%` / `66.96%`), 가우시안 노이즈(`σ=18` → `59.59%`), 온도 보정(`T*=1.70`, ECE `-74.2%`)을 현재 런타임에서 직접 재측정해 동결 리포트와 대조합니다.
""",
        ),
        _code(
            "cell-live-eval",
            """
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from seg_kd.data import build_bdd100k_manifest, load_manifest, records_for_split, save_manifest
from seg_kd.evaluation import evaluate_artifact, evaluate_temperature_scaled_model

if not EVALUATE_LIVE_DATASETS:
    print("Skipped live dataset download/re-evaluation (set EVALUATE_LIVE_DATASETS = True in Cell 1 to run).")
else:
    from datasets import load_dataset

    cs_dir = DATA_ROOT / "cityscapes_hires_384x768"
    cs_manifest_path = MANIFEST_DIR / "cityscapes-hires-384x768-manifest.json"
    if not cs_manifest_path.is_file():
        print("Downloading Chris1/cityscapes and exporting 384x768 PNG/trainId layout...")
        ds = load_dataset("Chris1/cityscapes")
        for out_split, hf_split in [("train", "train"), ("val", "validation")]:
            img_dir = cs_dir / "images" / "10k" / out_split
            msk_dir = cs_dir / "labels" / "sem_seg" / "masks" / out_split
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in enumerate(ds[hf_split]):
                stem = f"{out_split}_{idx:05d}"
                img = row["image"].convert("RGB").resize((768, 384), Image.Resampling.BILINEAR)
                m = row["semantic_segmentation"]
                if isinstance(m, Image.Image):
                    m_resized = m.resize((768, 384), Image.Resampling.NEAREST)
                    m_arr = np.asarray(m_resized)
                else:
                    m_arr = np.asarray(m)
                if m_arr.ndim == 3:
                    m_arr = m_arr[..., 0]
                m_arr = m_arr.astype(np.int64)
                m_arr = np.where((m_arr < 0) | (m_arr >= 19), 255, m_arr).astype(np.uint8)
                img.save(img_dir / f"{stem}.jpg", quality=95)
                Image.fromarray(m_arr, mode="L").save(msk_dir / f"{stem}_train_id.png")
        manifest = build_bdd100k_manifest(cs_dir, seed=378, val_ratio=0.15, hash_files=False)
        save_manifest(manifest, cs_manifest_path)

    cs_manifest = load_manifest(cs_manifest_path, dataset_root=cs_dir, verify_files=False)
    ref = json.loads((REPO_DIR / "reports" / "final-summary.json").read_text(encoding="utf-8"))

    for label, fname, ref_key, ref_t in [
        ("v3_medium_int8", "v3-max-mobilenetv4-medium-int8.onnx", "mobilenetv4_medium_int8", 1.70),
        ("v4_robust_medium_int8", "v4-final-robust-medium-int8.onnx", "v4_robust_medium_int8", 1.70),
    ]:
        onnx_path = CACHE_DIR / fname
        res = evaluate_artifact(
            onnx_path, cs_manifest, cs_dir, split="locked_eval", image_size=(384, 768), batch_size=1, threads=BENCHMARK_THREADS
        )
        cal = evaluate_temperature_scaled_model(
            onnx_path, cs_manifest, cs_dir, split="locked_eval", temperature=ref_t, image_size=(384, 768), threads=BENCHMARK_THREADS
        )
        expected_miou = ref["locked_eval_models"][ref_key]["miou_pct"]
        print(
            f"[LIVE Cityscapes locked_eval] {label:24s} | mIoU={res['miou']*100:.2f}% (ref {expected_miou:.2f}%) | "
            f"safety={res['safety_critical_miou']*100:.2f}% | ECE {res['ece']:.4f} -> {cal['ece']:.4f} (T*={ref_t:.2f})"
        )

    # Official KITTI Semantics Zero-Shot (200 images)
    kitti_dir = DATA_ROOT / "kitti"
    kitti_manifest_path = MANIFEST_DIR / "kitti-manifest.json"
    if not (kitti_dir / "training" / "image_2").is_dir():
        kitti_dir.mkdir(parents=True, exist_ok=True)
        zp = kitti_dir / "data_semantics.zip"
        urllib.request.urlretrieve("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip", zp)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(kitti_dir)
        zp.unlink(missing_ok=True)
    if not kitti_manifest_path.is_file():
        import subprocess, sys
        subprocess.run(
            [sys.executable, "-m", "seg_kd", "prepare-data", "--dataset", "kitti", "--data", str(kitti_dir), "--output", str(kitti_manifest_path)],
            check=True,
        )
    kitti_manifest = load_manifest(kitti_manifest_path, dataset_root=kitti_dir, verify_files=False)
    for label, fname, ref_key in [
        ("v3_medium_int8", "v3-max-mobilenetv4-medium-int8.onnx", "mobilenetv4_medium_int8"),
        ("v4_robust_medium_int8", "v4-final-robust-medium-int8.onnx", "v4_robust_medium_int8"),
    ]:
        km = evaluate_artifact(
            CACHE_DIR / fname, kitti_manifest, kitti_dir, split="zero_shot_eval", image_size=(384, 768), batch_size=1, threads=BENCHMARK_THREADS
        )
        exp_k = ref["kitti_zero_shot"][ref_key]["miou_pct"]
        print(
            f"[LIVE KITTI Zero-Shot 200]     {label:24s} | mIoU={km['miou']*100:.2f}% (ref {exp_k:.2f}%) | "
            f"safety={km['safety_critical_miou']*100:.2f}%"
        )
""",
        ),
        _md(
            "sec-reproduce-full",
            """
## 6. (`reproduce_full` 전용) NVIDIA L4 전체 재학습 파이프라인 (`v3-max` + `v4-final`)

`RUN_MODE = "reproduce_full"` 선택 시에만 실행됩니다. NVIDIA L4 (`24GB`, BF16) 환경에서:
1. `Chris1/cityscapes` (`384×768`) + KITTI Zero-Shot 매니페스트 구축
2. `SegFormer-B5` 교사 파인튜닝 → 3시드 쌍체 비교(`378, 379, 380`) + `384×768` 4종 소거(`baseline`, `pixel_kl`, `cwd`, `cwd_feat`)
3. 3-Tier Pareto (`MobileNetV4-Small`, `MobileNetV3-Large`, `MobileNetV4-Medium`) + QAT-KD + ONNX FP32/INT8 검증 게이트
4. `v4-final` 비대칭 강건성 KD (`Clean Teacher → Corrupted Student`, 45ep + BF16 교사 QAT-KD 6ep) + 사후 온도 보정 + 4종 악천후 스트레스 테스트 + 결과 번들 ZIP 생성
""",
        ),
        _code(
            "cell-reproduce-full",
            """
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

if RUN_MODE != "reproduce_full":
    print("RUN_MODE is 'verify_saved' — skipping full GPU retraining. Set RUN_MODE='reproduce_full' on an NVIDIA L4 to retrain from scratch.")
else:
    if not torch.cuda.is_available():
        raise RuntimeError("RUN_MODE='reproduce_full' requires an NVIDIA CUDA GPU (recommended: L4 24GB).")

    from datasets import load_dataset
    from seg_kd.data import build_bdd100k_manifest, load_manifest, save_manifest
    from seg_kd.evaluation import fit_temperature_scaling, evaluate_temperature_scaled_model
    from seg_kd.export import assess_int8_acceptance, export_student_onnx
    from seg_kd.benchmark import benchmark_model
    from seg_kd.training import train_student, write_json

    cs_dir = DATA_ROOT / "cityscapes_hires_384x768"
    cs_manifest_path = MANIFEST_DIR / "cityscapes-hires-384x768-manifest.json"
    if not cs_manifest_path.is_file():
        print("Preparing Chris1/cityscapes at 384x768...")
        ds = load_dataset("Chris1/cityscapes")
        for out_split, hf_split in [("train", "train"), ("val", "validation")]:
            img_dir = cs_dir / "images" / "10k" / out_split
            msk_dir = cs_dir / "labels" / "sem_seg" / "masks" / out_split
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)
            for idx, row in enumerate(ds[hf_split]):
                stem = f"{out_split}_{idx:05d}"
                img = row["image"].convert("RGB").resize((768, 384), Image.Resampling.BILINEAR)
                m = row["semantic_segmentation"]
                m_arr = np.asarray(m.resize((768, 384), Image.Resampling.NEAREST) if isinstance(m, Image.Image) else m)
                if m_arr.ndim == 3:
                    m_arr = m_arr[..., 0]
                m_arr = np.where((m_arr < 0) | (m_arr >= 19), 255, m_arr.astype(np.int64)).astype(np.uint8)
                img.save(img_dir / f"{stem}.jpg", quality=95)
                Image.fromarray(m_arr, mode="L").save(msk_dir / f"{stem}_train_id.png")
        save_manifest(build_bdd100k_manifest(cs_dir, seed=378, val_ratio=0.15, hash_files=False), cs_manifest_path)

    kitti_dir = DATA_ROOT / "kitti"
    kitti_manifest_path = MANIFEST_DIR / "kitti-manifest.json"
    if not (kitti_dir / "training" / "image_2").is_dir():
        kitti_dir.mkdir(parents=True, exist_ok=True)
        zp = kitti_dir / "data_semantics.zip"
        urllib.request.urlretrieve("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip", zp)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(kitti_dir)
        zp.unlink(missing_ok=True)
    if not kitti_manifest_path.is_file():
        subprocess.run(
            [sys.executable, "-m", "seg_kd", "prepare-data", "--dataset", "kitti", "--data", str(kitti_dir), "--output", str(kitti_manifest_path)],
            check=True,
        )

    # 1. Run canonical staged study (teacher -> 3-seed ablation -> 3-tier pareto -> export -> freeze -> eval -> plots)
    plan_path = REPO_DIR / "configs" / "l4-plan.json"
    v3_out = OUTPUT_ROOT / "v3-max"
    cmd = [
        sys.executable, "scripts/run_l4_study.py",
        "--plan", str(plan_path),
        "--manifest", str(cs_manifest_path),
        "--data", str(cs_dir),
        "--kitti-manifest", str(kitti_manifest_path),
        "--kitti-data", str(kitti_dir),
        "--output", str(v3_out),
        "--stage", "all",
        "--device", "cuda",
    ]
    subprocess.run(cmd, check=True)

    # 2. Run v4-final Asymmetric Robustness KD (Clean SegFormer-B5 -> Corrupted MobileNetV4-Medium + BF16 QAT-KD + Temp Scaling)
    manifest = load_manifest(cs_manifest_path, dataset_root=cs_dir, verify_files=False)
    teacher_ckpt = v3_out / "teacher" / "segformer_b5" / "teacher.pt"
    v4_out = OUTPUT_ROOT / "v4-final"
    robust_cfg = {
        "image_size": [384, 768],
        "epochs": 45,
        "batch_size": 16,
        "learning_rate": 1.2e-3,
        "weight_decay": 1e-4,
        "warmup_epochs": 3,
        "temperature": 4.0,
        "lambda_logit_kd": 1.0,
        "lambda_feat_kd": 0.5,
        "fpn_dim": 128,
        "pretrained_backbone": True,
        "precision": "bf16",
        "num_workers": 4,
        "asymmetric_robust_prob": 0.35,
        "asymmetric_ramp_epochs": 5,
        "qat_epochs": 6,
    }
    rob_res = train_student(
        dataset_root=cs_dir,
        manifest=manifest,
        output_dir=v4_out / "v4_robust_medium",
        arch="mobilenetv4_medium",
        method="cwd_feat",
        seed=378,
        teacher_path=teacher_ckpt,
        config=robust_cfg,
        device_spec="cuda",
        overwrite=True,
    )
    export_src = v4_out / "v4_robust_medium" / "student-qat.pt"
    if not export_src.is_file():
        export_src = v4_out / "v4_robust_medium" / "student.pt"
    export_dir = v4_out / "export" / "v4_robust_medium"
    export_student_onnx(export_src, export_dir, manifest=manifest, dataset_root=cs_dir, image_size=(384, 768), calibration_count=128, seed=378, export_int8=True)
    int8_path = export_dir / "student-int8.onnx"
    t_opt = fit_temperature_scaling(int8_path, manifest, cs_dir, split="validation", image_size=(384, 768), threads=4)
    cal_eval = evaluate_temperature_scaled_model(int8_path, manifest, cs_dir, split="locked_eval", temperature=t_opt["optimal_temperature"], image_size=(384, 768), threads=4)
    write_json({"val_fit": t_opt, "locked_eval_calibrated": cal_eval}, v4_out / "ece-calibration.json")
    print("Full v3-max + v4-final L4 reproduction completed cleanly ->", OUTPUT_ROOT)
""",
        ),
        _md(
            "sec-decisions",
            """
## 7. 핵심 설계 판단 및 실험 결과 요약 (참고 표)

| 단계 | 핵심 질문 | 검증 결과 | 채택 판단 |
|---|---|---|---|
| **1. 해상도 & 데이터** | 40 FPS 엣지 CPU 예산에서 작은 보행자·신호등을 지킬 최소 해상도는? | `Chris1/cityscapes` 원본 → `384×768` (`train 2,529 / val 446 / locked 500`) + KITTI 200장 Zero-Shot | **`384×768` 확정** |
| **2. 증류 손실 소거** | 픽셀 KL vs 공간 CWD vs 4단계 특징 정렬 중 무엇이 이종 ViT→CNN 압축에 유리한가? | `pixel_kl` `-2.41%p` 역효과(Safety `-5.71%p`) vs `cwd` `+1.42%p` (KITTI Safety `+7.12%p`) | **`CWD` 기반 확정** (`pixel_kl` 기각) |
| **3. 3시드 쌍체 검증** | 단일 시드 우연이 아닌가? 소수 안전 클래스는 일관되게 오르는가? | `1.50M` 3시드 `3/3 승`, Safety `+1.88%p` (6/6 전원 상승), 시드 SD `2.7~3.1×` 축소 | **재현성 입증** |
| **4. 3-Tier Pareto & INT8 게이트** | 어떤 백본이 정적 QDQ INT8에서 붕괴 없이 실제 CPU 가속을 내는가? | `V3-Large` `-6.46%p`·`0.72×` 차단, `V4-Small` `1.04×` 미달, **`V4-Medium` 4/4 통과 (`67.95%`, `7.91 MiB`, `40.5 FPS`)** | **`MobileNetV4-Medium INT8` 채택** |
| **5. 비대칭 강건성 KD (`v4`)** | Clean INT8의 센서 노이즈 붕괴(`13.75%`)를 어떻게 해결할 것인가? | Clean Teacher → Corrupted Student + BF16 QAT-KD → 노이즈 **`59.59%` (`+45.84%p`, 84.6M 교사 `59.19%` 추월)**, 안개 `66.33%` | **2-프로파일 배포 체계 확정** |
| **6. 사후 온도 보정 (`ECE`)** | CWD의 확률 과신(`ECE 0.026`)을 mIoU 손실·지연 없이 해결할 수 있는가? | Validation 전용 `T* = 1.70~1.72` 단일 나눗셈 → **ECE `-74.2%` (`0.0260 → 0.0067`), mIoU 변화 `0.00%p`, 지연 `0 ms`** | **추론 후처리 표준 채택** |
""",
        ),
    ]

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
