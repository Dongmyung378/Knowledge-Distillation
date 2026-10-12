import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from knowledge_distillation.runtime import (file_sha256, infer_image, load_class_map,
                                           load_model, preprocess_image, probabilities, validate_image_path)

HAS_TF = importlib.util.find_spec("tensorflow") is not None


class RuntimeValidationTests(unittest.TestCase):
    def test_stable_logits_softmax(self):
        logits = np.full((2, 43), -10_000.0)
        logits[:, 7] = 10_000.0
        actual = probabilities(logits)
        np.testing.assert_allclose(actual.sum(axis=1), 1)
        np.testing.assert_array_equal(actual.argmax(axis=1), [7, 7])
        with self.assertRaises(ValueError):
            probabilities(np.ones((1, 42)))
        with self.assertRaises(ValueError):
            probabilities(np.full((1, 43), np.nan))

    def test_map_requires_complete_43_ids(self):
        self.assertEqual(load_class_map()[42], "42")
        supplied = {str(i): "class-" + str(i) for i in range(43)}
        self.assertEqual(load_class_map(supplied)[42], "class-42")
        with self.assertRaises(ValueError):
            load_class_map(["incomplete"])

    def test_model_checksum_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "fake.keras"
            model.write_bytes(b"not a real model")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_model(model, expected_sha256="0" * 64)
            invalid = Path(temporary) / "text.txt"
            invalid.write_text("not an image")
            with self.assertRaises(ValueError):
                validate_image_path(invalid)


@unittest.skipUnless(HAS_TF, "TensorFlow runtime is not installed")
class SavedRuntimeTests(unittest.TestCase):
    def test_saved_keras_png_prediction_roundtrip(self):
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = tf.keras.Input((96, 96, 3))
            output = tf.keras.layers.Dense(43, kernel_initializer="zeros")(tf.keras.layers.GlobalAveragePooling2D()(inputs))
            model = tf.keras.Model(inputs, output)
            biases = np.zeros(43, np.float32)
            biases[9] = 3.0
            model.layers[-1].set_weights([np.zeros((3, 43), np.float32), biases])
            model_path = root / "tiny.keras"
            model.save(model_path)
            image = root / "image.png"
            image.write_bytes(tf.io.encode_png(np.full((11, 13, 3), 150, np.uint8)).numpy())
            result = infer_image(model_path, image, expected_sha256=file_sha256(model_path))
            self.assertEqual(result["predicted_class_id"], 9)
            self.assertEqual(result["class_name"], "9")
            self.assertEqual(len(result["class_names"]), 43)
            self.assertEqual(result["image_sha256"], file_sha256(image))
            json.dumps(result, allow_nan=False)
            from PIL import Image
            ppm = root / "image.ppm"
            Image.fromarray(np.full((11, 13, 3), 150, np.uint8)).save(ppm)
            np.testing.assert_allclose(preprocess_image(image), preprocess_image(ppm), atol=0)

    def test_safe_mode_refuses_lambda_layer(self):
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as temporary:
            inputs = tf.keras.Input((96, 96, 3))
            value = tf.keras.layers.Lambda(lambda x: x)(inputs)
            output = tf.keras.layers.Dense(43)(tf.keras.layers.GlobalAveragePooling2D()(value))
            path = Path(temporary) / "unsafe.keras"
            tf.keras.Model(inputs, output).save(path)
            with self.assertRaises(ValueError):
                load_model(path)
