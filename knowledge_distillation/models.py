"""Plain Keras models with explicit pretrained-input and BN policies."""

from __future__ import annotations

import tensorflow as tf


def build_augmentation(seed: int = 378):
    # Never flip traffic signs: left/right direction changes their class.
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0), seed=seed),
            tf.keras.layers.RandomContrast(0.2, value_range=(0.0, 1.0), seed=seed + 1),
            tf.keras.layers.RandomZoom(0.05, 0.05, seed=seed + 2),
        ],
        name="safe_augmentation",
    )


def build_student(image_size: int = 96, num_classes: int = 43):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(image_size, image_size, 3)),
            tf.keras.layers.SeparableConv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.SeparableConv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes),
        ],
        name="Student_MicroCNN",
    )
    if num_classes == 43 and model.count_params() != 34_406:
        raise AssertionError("The historical compact architecture must retain 34,406 parameters")
    return model


def build_teacher(
    image_size: int = 96,
    num_classes: int = 43,
    *,
    variant: str = "frozen_bn",
    seed: int = 378,
    weights: str | None = "imagenet",
):
    if variant not in {"frozen_bn", "adaptive_bn"}:
        raise ValueError("Teacher variant must be frozen_bn or adaptive_bn")
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = build_augmentation(seed)(inputs)
    # Dataset contract is [0, 1]; pretrained MobileNetV2 expects [-1, 1].
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenet_input_scaling")(x)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights=weights
    )
    base.trainable = False
    x = base(x, training=False) if variant == "frozen_bn" else base(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    logits = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs, logits, name=f"Teacher_MobileNetV2_{variant}")


def configure_teacher_stage(model, stage: str, *, variant: str = "frozen_bn", unfreeze_last: int = 60) -> None:
    if stage not in {"head", "fine_tune"} or variant not in {"frozen_bn", "adaptive_bn"}:
        raise ValueError("Unknown teacher stage or BN policy")
    base = next((layer for layer in model.layers if layer.name.startswith("mobilenetv2")), None)
    if base is None:
        raise ValueError("Teacher model has no MobileNetV2 backbone")
    model.trainable = True
    base.trainable = stage == "fine_tune"
    if stage == "fine_tune":
        for index, layer in enumerate(base.layers):
            layer.trainable = index >= len(base.layers) - unfreeze_last
            if variant == "frozen_bn" and isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False


def weights_sha256(model) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in model.get_weights():
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()
