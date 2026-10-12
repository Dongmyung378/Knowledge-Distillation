"""Safe, CPU-capable inference for plain Keras and TFLite classifiers.

All exported models consume RGB float32 images in [0, 1]. The teacher's
MobileNetV2 preprocessing is part of its saved graph, not applied a second time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_SIZE = 96
NUM_CLASSES = 43
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensorflow():
    # This module imports without TensorFlow, allowing a useful installation
    # error before the optional framework is installed.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("Install the project's pinned TensorFlow runtime first.") from exc
    return tf


def configure_cpu(threads: int = 1) -> dict[str, Any]:
    """Set CPU-only execution before TensorFlow creates its first context."""
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    tf = tensorflow()
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError("CPU configuration must precede GPU initialization.") from exc
    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError as exc:
        current = tf.config.threading.get_intra_op_parallelism_threads()
        inter = tf.config.threading.get_inter_op_parallelism_threads()
        if current != threads or inter != 1:
            raise RuntimeError("Start a fresh process to set benchmark thread counts.") from exc
    return {"device": "CPU", "intra_op_threads": threads, "inter_op_threads": 1,
            "tensorflow_version": tf.__version__}


def validate_image_path(path: str | Path) -> Path:
    image = Path(path).expanduser().resolve(strict=True)
    if not image.is_file() or image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}:
        raise ValueError("Provide an existing PNG, JPEG, BMP or PPM image file.")
    size = image.stat().st_size
    if size == 0 or size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image must contain 1 to {MAX_IMAGE_BYTES} bytes.")
    return image


def preprocess_image(path: str | Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Use the same bilinear RGB [0,1] pipeline as evaluation and training."""
    image = validate_image_path(path)
    tf = tensorflow()
    if image.suffix.lower() == ".ppm":
        from PIL import Image
        with Image.open(image) as opened:
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions exceed the supported pixel limit.")
            decoded = tf.convert_to_tensor(np.asarray(opened.convert("RGB"), dtype=np.uint8))
    else:
        raw = tf.io.read_file(str(image))
        decoded = tf.io.decode_image(raw, channels=3, expand_animations=False)
    dimensions = tf.shape(decoded).numpy()
    if len(dimensions) != 3 or int(dimensions[0]) * int(dimensions[1]) > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions exceed the supported pixel limit.")
    value = tf.image.resize(tf.cast(decoded, tf.float32), (image_size, image_size)) / 255.0
    return value.numpy().astype(np.float32, copy=False)


def load_class_map(value: str | Path | dict | list | None = None) -> list[str]:
    """Prefer dataset IDs unless a complete, explicitly supplied map is available."""
    if value is None:
        return [str(i) for i in range(NUM_CLASSES)]
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("class_names", value.get("class_map", value))
        if isinstance(value, dict):
            value = [value.get(str(i), value.get(i)) for i in range(NUM_CLASSES)]
    if not isinstance(value, list) or len(value) != NUM_CLASSES or any(
        not isinstance(name, str) or not name.strip() for name in value
    ):
        raise ValueError("Class map must explicitly provide all 43 class IDs.")
    return value


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES or not np.isfinite(logits).all():
        raise ValueError("Classifier must return finite [batch,43] logits.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


class KerasRunner:
    def __init__(self, path: Path):
        tf = tensorflow()
        self.model = tf.keras.models.load_model(str(path), compile=False, safe_mode=True)
        shape = self.model.input_shape
        output = self.model.output_shape
        if isinstance(shape, list) or tuple(shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError("Model must have one [batch,96,96,3] input.")
        if isinstance(output, list) or tuple(output[1:]) != (NUM_CLASSES,):
            raise ValueError("Model must have one [batch,43] logits output.")
        self.parameters = int(self.model.count_params())
        # A traced inference function avoids model.predict's dataset machinery.
        self._call = tf.function(lambda x: self.model(x, training=False),
                                 input_signature=[tf.TensorSpec([None, IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32)])

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self._call(np.asarray(images, dtype=np.float32)).numpy()


class TFLiteRunner:
    def __init__(self, path: Path, threads: int = 1):
        tf = tensorflow()
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("TFLite model requires exactly one input and output.")
        self.input = inputs[0]
        self.output = outputs[0]
        if tuple(self.input["shape"][1:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError("TFLite model must accept [batch,96,96,3].")
        if tuple(self.output["shape"][1:]) != (NUM_CLASSES,):
            raise ValueError("TFLite model must output [batch,43] logits.")
        self.parameters = None  # File format does not expose Keras parameter count.

    @staticmethod
    def _quantize(value: np.ndarray, detail: dict) -> np.ndarray:
        dtype = detail["dtype"]
        if not np.issubdtype(dtype, np.integer):
            return value.astype(dtype)
        scale, zero_point = detail["quantization"]
        if scale <= 0:
            raise ValueError("Integer tensor lacks a valid quantization scale.")
        limits = np.iinfo(dtype)
        return np.clip(np.rint(value / scale + zero_point), limits.min, limits.max).astype(dtype)

    def predict(self, images: np.ndarray) -> np.ndarray:
        # Fixed batch-one export works on any caller batch without reallocating.
        results = []
        for image in np.asarray(images, dtype=np.float32):
            value = self._quantize(image[None], self.input)
            self.interpreter.set_tensor(self.input["index"], value)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output["index"])
            if np.issubdtype(self.output["dtype"], np.integer):
                scale, zero_point = self.output["quantization"]
                output = (output.astype(np.float32) - zero_point) * scale
            results.append(output[0])
        return np.asarray(results, dtype=np.float32).reshape((-1, NUM_CLASSES))


def load_model(path: str | Path, expected_sha256: str | None = None, threads: int = 1):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in {".keras", ".tflite"}:
        raise ValueError("Only plain .keras and .tflite model files are supported.")
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError("Model SHA-256 does not match the expected artifact.")
    return TFLiteRunner(path, threads=threads) if path.suffix == ".tflite" else KerasRunner(path)


def infer_image(model_path: str | Path, image_path: str | Path,
                class_map=None, expected_sha256: str | None = None,
                top_k: int = 5, threads: int = 1) -> dict[str, Any]:
    if not 1 <= top_k <= NUM_CLASSES:
        raise ValueError("top_k must be between 1 and 43")
    classes = load_class_map(class_map)
    image = preprocess_image(image_path)
    runner = load_model(model_path, expected_sha256=expected_sha256, threads=threads)
    scores = probabilities(runner.predict(image[None]))[0]
    order = np.argsort(-scores, kind="stable")[:top_k]
    return {"schema_version": 1, "model_sha256": file_sha256(model_path),
            "image_sha256": file_sha256(image_path), "predicted_class_id": int(order[0]),
            "class_name": classes[int(order[0])], "class_names": classes,
            "top_predictions": [{"class_id": int(i), "class_name": classes[int(i)],
                                 "probability": float(scores[i])} for i in order],
            "probability_note": "Softmax scores are not calibrated reliability estimates.",
            "preprocessing": {"image_size": [IMAGE_SIZE, IMAGE_SIZE], "channels": "RGB",
                              "external_range": [0, 1], "resize": "bilinear",
                              "teacher_scaling": "Saved inside the teacher model graph."}}
