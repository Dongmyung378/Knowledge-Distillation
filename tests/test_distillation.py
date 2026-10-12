import importlib.util
import unittest

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
if HAS_TENSORFLOW:
    import numpy as np
    import tensorflow as tf

    from knowledge_distillation.losses import Distiller, distillation_terms
    from knowledge_distillation.models import build_student, build_teacher, configure_teacher_stage, weights_sha256


@unittest.skipUnless(HAS_TENSORFLOW, "TensorFlow is required for optimizer correctness tests")
class DistillationCorrectnessTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(378)

    def test_historical_student_parameter_budget_is_preserved(self):
        with tf.device("/CPU:0"):
            model = build_student()
        self.assertEqual(model.count_params(), 34_406)
        self.assertEqual(model.output_shape[-1], 43)

    def test_teacher_preprocessing_and_fine_tuning_keep_bn_frozen(self):
        with tf.device("/CPU:0"):
            teacher = build_teacher(image_size=32, num_classes=2, weights=None)
            preprocessing = tf.keras.Model(teacher.input, teacher.get_layer("mobilenet_input_scaling").output)
            black_white = tf.constant(np.stack([np.zeros((32, 32, 3)), np.ones((32, 32, 3))]), tf.float32)
            scaled = preprocessing(black_white, training=False).numpy()
            np.testing.assert_allclose(scaled[0], -1.0)
            np.testing.assert_allclose(scaled[1], 1.0)
            configure_teacher_stage(teacher, "head")
            base = next(layer for layer in teacher.layers if layer.name.startswith("mobilenetv2"))
            self.assertEqual(len(base.trainable_variables), 0)
            configure_teacher_stage(teacher, "fine_tune")
            self.assertGreater(len(base.trainable_variables), 0)
            self.assertTrue(all(not layer.trainable for layer in base.layers if isinstance(layer, tf.keras.layers.BatchNormalization)))

    def test_alpha_one_matches_weighted_ce_value_and_gradients(self):
        with tf.device("/CPU:0"):
            logits = tf.Variable([[0.8, -1.2, 0.1], [-0.3, 0.2, 1.4]], dtype=tf.float32)
            labels = tf.constant([0, 2])
            weights = tf.constant([0.5, 2.0])
            with tf.GradientTape() as kd_tape:
                kd = distillation_terms(labels, logits, alpha=1.0, sample_weight=weights)["loss"]
            with tf.GradientTape() as ce_tape:
                ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits, sample_weight=weights)
            np.testing.assert_allclose(kd.numpy(), ce.numpy(), atol=1e-7)
            np.testing.assert_allclose(kd_tape.gradient(kd, logits).numpy(), ce_tape.gradient(ce, logits).numpy(), atol=1e-7)

    def test_equal_teacher_student_logits_have_zero_kl(self):
        logits = tf.constant([[10.0, 2.0, -3.0], [0.0, 0.5, -1.0]])
        terms = distillation_terms(tf.constant([0, 1]), logits, logits, temperature=10.0)
        self.assertAlmostEqual(float(terms["soft_loss"].numpy()), 0.0, places=7)

    def test_temperature_compensation_and_unweighted_kl_are_explicit(self):
        labels = tf.constant([0, 1])
        student = tf.constant([[1.0, -1.0], [-2.0, 2.0]])
        teacher = tf.constant([[2.0, -2.0], [0.5, -0.5]])
        a = distillation_terms(labels, student, teacher, alpha=0.3, temperature=5.0)
        b = distillation_terms(labels, student, teacher, alpha=0.3, temperature=5.0, sample_weight=[3.0, 0.2])
        np.testing.assert_allclose(a["soft_loss"].numpy(), b["soft_loss"].numpy(), atol=1e-8)
        np.testing.assert_allclose(a["loss"].numpy(), 0.3 * a["hard_loss"].numpy() + 0.7 * 25 * a["soft_loss"].numpy(), atol=1e-7)

    def test_teacher_weights_immutable_after_actual_optimizer_step(self):
        with tf.device("/CPU:0"):
            teacher = tf.keras.Sequential([tf.keras.layers.Input((4,)), tf.keras.layers.Dense(3)])
            student = tf.keras.Sequential([tf.keras.layers.Input((4,)), tf.keras.layers.Dense(3)])
            trainer = Distiller(student, teacher, alpha=0.3, temperature=3.0, class_weights={0: 0.5, 1: 2.0, 2: 1.0})
            trainer.compile(optimizer=tf.keras.optimizers.Adam(0.01), run_eagerly=True)
            before_teacher = weights_sha256(teacher)
            before_student = weights_sha256(student)
            scores = trainer.train_step((tf.constant([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]), tf.constant([0, 1])))
            self.assertEqual(weights_sha256(teacher), before_teacher)
            self.assertNotEqual(weights_sha256(student), before_student)
            self.assertTrue(np.isfinite(float(scores["loss"].numpy())))

    def test_validation_loss_is_true_unweighted_ce_not_zero(self):
        with tf.device("/CPU:0"):
            student = tf.keras.Sequential([tf.keras.layers.Input((2,)), tf.keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros")])
            trainer = Distiller(student, alpha=1.0, class_weights={0: 20.0, 1: 0.1})
            trainer.reset_metrics()
            scores = trainer.test_step((tf.constant([[1.0, 2.0], [3.0, 4.0]]), tf.constant([0, 1])))
            self.assertAlmostEqual(float(scores["loss"].numpy()), np.log(2), places=6)


if __name__ == "__main__":
    unittest.main()
