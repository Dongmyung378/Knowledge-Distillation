"""Stable logits-based distillation and meaningful validation CE metrics."""

from __future__ import annotations

import tensorflow as tf


def distillation_terms(
    labels,
    student_logits,
    teacher_logits=None,
    *,
    alpha: float = 0.3,
    temperature: float = 5.0,
    sample_weight=None,
):
    """Weighted hard CE plus unweighted Teacher-to-Student KL times T squared.

    Weights multiply CE per example, then divide by batch size (Keras
    SUM_OVER_BATCH_SIZE semantics). Soft KL deliberately has no class weights.
    ``alpha=1`` bypasses teacher computation and equals supervised CE exactly.
    """
    if not 0.0 <= alpha <= 1.0 or temperature <= 0:
        raise ValueError("alpha must be in [0, 1] and temperature must be positive")
    student_logits = tf.cast(student_logits, tf.float32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=student_logits)
    if sample_weight is not None:
        ce *= tf.cast(sample_weight, tf.float32)
    hard_loss = tf.reduce_mean(ce)
    if alpha == 1.0:
        soft_loss = tf.constant(0.0, tf.float32)
    else:
        if teacher_logits is None:
            raise ValueError("Teacher logits are required for alpha below one")
        t_log_prob = tf.nn.log_softmax(tf.stop_gradient(tf.cast(teacher_logits, tf.float32)) / temperature, axis=-1)
        s_log_prob = tf.nn.log_softmax(student_logits / temperature, axis=-1)
        soft_loss = tf.reduce_mean(tf.reduce_sum(tf.exp(t_log_prob) * (t_log_prob - s_log_prob), axis=-1))
    return {
        "hard_loss": hard_loss,
        "soft_loss": soft_loss,
        "loss": alpha * hard_loss + (1.0 - alpha) * temperature**2 * soft_loss,
    }


class Distiller(tf.keras.Model):
    """Training wrapper only; the saved deployment model is its plain student."""

    def __init__(
        self,
        student,
        teacher=None,
        *,
        alpha: float = 0.3,
        temperature: float = 5.0,
        class_weights=None,
        augmentation=None,
    ):
        super().__init__(name="DistillationTrainer")
        if not 0 <= alpha <= 1 or temperature <= 0:
            raise ValueError("Invalid distillation hyperparameters")
        if alpha < 1 and teacher is None:
            raise ValueError("Distillation requires a teacher")
        self.student = student
        self.teacher = teacher
        if teacher is not None:
            teacher.trainable = False
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.augmentation = augmentation
        self.class_weight_vector = None
        if class_weights is not None:
            n = int(student.output_shape[-1])
            self.class_weight_vector = tf.constant([class_weights.get(i, class_weights.get(str(i), 1.0)) for i in range(n)], tf.float32)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.hard_loss_tracker = tf.keras.metrics.Mean(name="hard_loss")
        self.soft_loss_tracker = tf.keras.metrics.Mean(name="soft_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.hard_loss_tracker, self.soft_loss_tracker, self.accuracy_tracker]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def train_step(self, data):
        x, labels, supplied_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        if self.augmentation is not None:
            x = self.augmentation(x, training=True)
            x = tf.clip_by_value(x, 0.0, 1.0)
        weight = supplied_weight
        if self.class_weight_vector is not None:
            class_weight = tf.gather(self.class_weight_vector, tf.cast(labels, tf.int32))
            weight = class_weight if weight is None else class_weight * tf.cast(weight, tf.float32)
        teacher_logits = self.teacher(x, training=False) if self.alpha < 1 else None
        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            terms = distillation_terms(
                labels, student_logits, teacher_logits, alpha=self.alpha,
                temperature=self.temperature, sample_weight=weight,
            )
            regularization = tf.add_n(self.student.losses) if self.student.losses else tf.constant(0.0, tf.float32)
            total_loss = terms["loss"] + regularization
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        batch_size = tf.cast(tf.shape(labels)[0], tf.float32)
        self.loss_tracker.update_state(total_loss, sample_weight=batch_size)
        self.hard_loss_tracker.update_state(terms["hard_loss"], sample_weight=batch_size)
        self.soft_loss_tracker.update_state(terms["soft_loss"], sample_weight=batch_size)
        # Public accuracy is unweighted for both baseline and distilled student.
        self.accuracy_tracker.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x, labels, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        logits = self.student(x, training=False)
        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32), logits=tf.cast(logits, tf.float32)))
        size = tf.cast(tf.shape(labels)[0], tf.float32)
        # Validation loss is real, unweighted hard-label CE, never an untouched
        # metric whose reset value would be zero.
        self.loss_tracker.update_state(ce, sample_weight=size)
        self.hard_loss_tracker.update_state(ce, sample_weight=size)
        self.soft_loss_tracker.update_state(0.0, sample_weight=size)
        self.accuracy_tracker.update_state(labels, logits)
        return {metric.name: metric.result() for metric in self.metrics}
