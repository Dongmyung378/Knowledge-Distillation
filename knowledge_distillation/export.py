"""FP32 and full integer TFLite export, calibrated on training records only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .evaluation import (canonical_sha256, read_manifest, records_for_evaluation,
                         validate_freeze, verify_record_files)
from .runtime import file_sha256, load_model, tensorflow


def calibration_records(manifest: dict, count: int = 128, seed: int = 378,
                        input_split: str = "train") -> list[dict]:
    if input_split != "train" or manifest.get("external_audit"):
        raise ValueError("INT8 calibration is permitted on training records only.")
    if not isinstance(count, int) or count < 1:
        raise ValueError("Calibration count must be a positive integer.")
    candidates = records_for_evaluation(manifest, "train")
    rng = np.random.default_rng(seed)
    # Round-robin class coverage instead of a fortuitous unstratified sample.
    buckets = {}
    for record in sorted(candidates, key=lambda r: r["id"]):
        buckets.setdefault(record["label"], []).append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    labels = list(sorted(buckets))
    rng.shuffle(labels)
    selected = []
    while len(selected) < min(count, len(candidates)):
        for label in labels:
            if buckets[label]:
                selected.append(buckets[label].pop())
                if len(selected) == min(count, len(candidates)):
                    break
    if any(r["split"] != "train" for r in selected):
        raise ValueError("Calibration leakage: a nontraining record was selected.")
    return selected


def export_student(model_path, manifest, dataset_root, output_dir,
                   calibration_count: int = 128, seed: int = 378,
                   input_split: str = "train", freeze_path=None) -> dict:
    manifest = read_manifest(manifest)
    if freeze_path is not None:
        validate_freeze(freeze_path, manifest, model_path)
    chosen = calibration_records(manifest, calibration_count, seed, input_split)
    verify_record_files(chosen, dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runner = load_model(model_path)
    if not hasattr(runner, "model"):
        raise ValueError("Export requires a plain .keras student model.")
    model = runner.model
    tf = tensorflow()
    from .data import make_dataset

    def representative_dataset():
        for image, _ in make_dataset(chosen, dataset_root, training=False,
                                     seed=seed, batch_size=1):
            yield [np.asarray(image, dtype=np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    fp32_content = converter.convert()
    fp32_path = output / "student-fp32.tflite"
    fp32_path.write_bytes(fp32_content)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_content = converter.convert()
    int8_path = output / "student-int8.tflite"
    int8_path.write_bytes(int8_content)
    fp32 = load_model(fp32_path)
    int8 = load_model(int8_path)
    if int8.input["dtype"] != np.int8 or int8.output["dtype"] != np.int8:
        raise ValueError("The integer export did not produce INT8 input/output tensors.")
    float_tensor_names = [d["name"] for d in int8.interpreter.get_tensor_details()
                          if np.issubdtype(d["dtype"], np.floating)]
    if float_tensor_names:
        raise ValueError("Full INT8 export unexpectedly contains floating-point tensors.")
    samples = chosen[:min(16, len(chosen))]
    max_absolute_diff = 0.0
    fp32_agreement = 0
    int8_agreement = 0
    for images, _ in make_dataset(samples, dataset_root, training=False, seed=seed, batch_size=1):
        images = np.asarray(images, dtype=np.float32)
        keras_logits = runner.predict(images)
        fp32_logits = fp32.predict(images)
        int8_logits = int8.predict(images)
        if not (np.isfinite(fp32_logits).all() and np.isfinite(int8_logits).all()):
            raise ValueError("Reloaded exported model produced nonfinite logits.")
        if not np.allclose(keras_logits, fp32_logits, rtol=1e-4, atol=1e-4):
            raise ValueError("FP32 TFLite forward pass differs materially from saved Keras model.")
        max_absolute_diff = max(max_absolute_diff, float(np.max(np.abs(keras_logits - fp32_logits))))
        fp32_agreement += int(keras_logits.argmax() == fp32_logits.argmax())
        int8_agreement += int(keras_logits.argmax() == int8_logits.argmax())
    calibration = {"schema_version": 1, "split": "train", "seed": seed,
                   "requested_count": calibration_count, "count": len(chosen),
                   "class_ids": sorted({r["label"] for r in chosen}),
                   "selection": "Seeded class-stratified round robin, without augmentation",
                   "records": [{"id": r["id"], "label": r["label"], "sha256": r["sha256"],
                                "split": r["split"]} for r in chosen],
                   "manifest_sha256": canonical_sha256(manifest),
                   "selected_records_sha256": canonical_sha256(chosen)}
    (output / "calibration-manifest.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    report = {"schema_version": 1, "source_model_sha256": file_sha256(model_path),
              "source_model_parameters": runner.parameters,
              "tensorflow_version": tf.__version__, "calibration_count": len(chosen),
              "calibration_manifest_sha256": file_sha256(output / "calibration-manifest.json"),
              "calibration_split": "train", "fp32_forward_checked_count": len(samples),
              "fp32_max_absolute_logit_difference": max_absolute_diff,
              "fp32_argmax_agreement_count": fp32_agreement,
              "int8_argmax_agreement_on_training_smoke_count": int8_agreement,
              "smoke_scope": "Training calibration subset only, not held-out accuracy.",
              "exports": {"fp32": {"filename": fp32_path.name, "bytes": len(fp32_content),
                                     "sha256": file_sha256(fp32_path), "input_dtype": "float32"},
                          "int8": {"filename": int8_path.name, "bytes": len(int8_content),
                                     "sha256": file_sha256(int8_path), "input_dtype": "int8",
                                     "output_dtype": "int8", "floating_tensor_count": 0}},
              "acceptance_policy": {"max_accuracy_drop_pp": 1.0,
                                    "requires_smaller_file_than_fp32": True,
                                    "minimum_same_cpu_p50_speedup": 1.05,
                                    "selection_split": "validation"},
              "deployment_selection": "pending validation accuracy and same-CPU latency measurements; final evaluation does not select quantization"}
    (output / "export-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def assess_int8_acceptance(fp32_metrics: dict, int8_metrics: dict,
                           fp32_benchmark: dict, int8_benchmark: dict,
                           max_accuracy_drop_pp: float = 1.0,
                           minimum_speedup: float = 1.05) -> dict:
    """A smaller file alone does not establish an effective deployment win."""
    if not 0 <= max_accuracy_drop_pp <= 1.0:
        raise ValueError("Declared INT8 accuracy-loss allowance must be at most 1 percentage point.")
    if not np.isfinite(minimum_speedup) or minimum_speedup < 1.05:
        raise ValueError("Declared INT8 speedup allowance must require at least 1.05x CPU speedup.")
    if fp32_metrics.get("split") != "validation" or int8_metrics.get("split") != "validation":
        raise ValueError("Quantization selection uses validation only, never final holdout or official test outcomes.")
    bindings = ("split", "evaluation_records_sha256", "freeze_sha256")
    if any(fp32_metrics.get(k) != int8_metrics.get(k) for k in bindings):
        raise ValueError("INT8 accuracy must use the same frozen images and evaluation policy.")
    for metrics, benchmark in ((fp32_metrics, fp32_benchmark), (int8_metrics, int8_benchmark)):
        if not metrics.get("model_sha256") or metrics["model_sha256"] != benchmark.get("model_sha256"):
            raise ValueError("Accuracy and latency must refer to the exact same model artifact.")
        if not metrics.get("training_manifest_sha256") or metrics["training_manifest_sha256"] != benchmark.get("manifest_sha256"):
            raise ValueError("Accuracy and latency must bind the same original dataset inventory.")
    for key in ("sample_ids_sha256", "threads", "repetitions", "warmup", "device"):
        if fp32_benchmark.get(key) != int8_benchmark.get(key):
            raise ValueError("INT8 latency comparison must use identical CPU benchmark conditions.")
    if fp32_benchmark.get("device") != "CPU":
        raise ValueError("INT8 acceptance requires the declared CPU benchmark.")
    accuracy_drop_pp = 100.0 * (fp32_metrics["accuracy"] - int8_metrics["accuracy"])
    fp32_latency = fp32_benchmark["pure_model"]["p50_ms"]
    int8_latency = int8_benchmark["pure_model"]["p50_ms"]
    if not all(np.isfinite(value) and value > 0 for value in (fp32_latency, int8_latency)):
        raise ValueError("Measured p50 latency must be finite and positive.")
    speedup = fp32_latency / int8_latency
    checks = {"accuracy_drop_within_limit": accuracy_drop_pp <= max_accuracy_drop_pp,
              "file_size_improved": int8_benchmark["model_bytes"] < fp32_benchmark["model_bytes"],
              "same_cpu_p50_speedup_within_limit": speedup >= minimum_speedup}
    return {"accepted": all(checks.values()), "checks": checks,
            "accuracy_drop_pp": accuracy_drop_pp, "max_accuracy_drop_pp": max_accuracy_drop_pp,
            "same_cpu_p50_speedup": speedup, "minimum_speedup": minimum_speedup,
            "file_size_ratio_int8_over_fp32": int8_benchmark["model_bytes"] / fp32_benchmark["model_bytes"],
            "latency_ratio_int8_over_fp32": int8_latency / fp32_latency,
            "selected_format": "INT8" if all(checks.values()) else "FP32",
            "limitation": "One host and fixed thread count; repeat measurements on the target device."}
