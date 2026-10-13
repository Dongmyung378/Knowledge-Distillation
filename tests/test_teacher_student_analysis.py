"""Tiny synthetic safety/counting tests; no model execution or quality claims."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('teacher_analysis', Path(__file__).parents[1] / 'scripts/analyze_teacher_student.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class SavedPredictionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.records = [{'id': str(i), 'label': i % 2, 'track_id': f't{i // 2}', 'split': 'locked_holdout'} for i in range(4)]
        self.reference = [{'id': r['id'], 'label': r['label'], 'prediction': p, 'confidence': .9}
                          for r, p in zip(self.records, (0, 0, 0, 0))]
        self.candidate = [{'id': r['id'], 'label': r['label'], 'prediction': p, 'confidence': .8}
                          for r, p in zip(self.records, (0, 1, 1, 1))]

    def test_paired_repairs_regressions_and_track_resampling_are_exact(self):
        result, _, classes, tracks = analysis.compare_decisions(self.reference, self.candidate, self.records, draws=80, seed=17)
        self.assertEqual(result['reference_wrong_candidate_correct'], 2)
        self.assertEqual(result['reference_correct_candidate_wrong'], 1)
        self.assertEqual(result['net_correct'], 1)
        self.assertEqual(result['accuracy_gain_pp'], 25)
        self.assertEqual(sum(row['support'] for row in classes), 4)
        self.assertEqual(len(classes), 43)
        self.assertEqual(len(tracks), 2)
        self.assertEqual(result['track_cluster_bootstrap'], analysis.track_bootstrap(tracks, draws=80, seed=17))

    def test_hash_bound_prediction_order_and_label_changes_are_rejected(self):
        manifest = {'class_names': [str(i) for i in range(43)]}
        freeze = {'freeze_sha256': 'freeze', 'models': {'model': {'sha256': 'weight'}}}
        observed, _ = analysis.metrics_from_decisions(self.reference)
        metric = {**observed, 'split': 'locked_holdout', 'selection_permitted': False,
                  'manifest_sha256': analysis.canonical_sha(manifest), 'training_manifest_sha256': analysis.canonical_sha(manifest),
                  'evaluation_records_sha256': analysis.canonical_sha(self.records), 'freeze_sha256': 'freeze',
                  'class_names': manifest['class_names'], 'model_sha256': 'weight',
                  'preprocessing': {'image_size': [96, 96], 'channels': 'RGB', 'range': [0, 1]}}
        analysis.checked_predictions(self.reference, metric, self.records, manifest, freeze)
        for kind in ('order', 'label', 'count', 'hash'):
            rows, m = copy.deepcopy(self.reference), copy.deepcopy(metric)
            if kind == 'order': rows[0], rows[1] = rows[1], rows[0]
            if kind == 'label': rows[0]['label'] = 1
            if kind == 'count': rows.pop()
            if kind == 'hash': m['evaluation_records_sha256'] = 'different'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                analysis.checked_predictions(rows, m, self.records, manifest, freeze)

    def test_changed_prediction_hash_and_nonfinite_confidence_are_rejected(self):
        manifest = {'class_names': [str(i) for i in range(43)]}
        freeze = {'freeze_sha256': 'freeze', 'models': {'model': {'sha256': 'weight'}}}
        observed, _ = analysis.metrics_from_decisions(self.reference)
        metric = {**observed, 'split': 'locked_holdout', 'selection_permitted': False,
                  'manifest_sha256': analysis.canonical_sha(manifest), 'training_manifest_sha256': analysis.canonical_sha(manifest),
                  'evaluation_records_sha256': analysis.canonical_sha(self.records), 'freeze_sha256': 'freeze',
                  'class_names': manifest['class_names'], 'model_sha256': 'weight',
                  'preprocessing': {'image_size': [96, 96], 'channels': 'RGB', 'range': [0, 1]}}
        with self.assertRaises(ValueError):
            analysis.checked_predictions(self.reference, metric, self.records, manifest, freeze,
                                         expected_prediction_sha='original', actual_prediction_sha='changed')
        rows = copy.deepcopy(self.reference)
        rows[0]['confidence'] = float('nan')
        with self.assertRaises(ValueError):
            analysis.checked_predictions(rows, metric, self.records, manifest, freeze)

    def test_soft_control_init_or_teacher_difference_cannot_be_called_matched(self):
        base = {'seed': 378, 'configuration': dict(max_epochs=30, patience=5, batch_size=64,
                learning_rate=.001, augmentation=True, deterministic=True, alpha=1, temperature=5),
                'initial_weights_sha256': 'same', 'initial_weights_file_sha256': 'file',
                'teacher_file_sha256': 'teacher', 'teacher_immutable': True, 'epochs_run': 30,
                'learning_rate_history': [.001] * 30}
        models = {'teacher': {'metrics': {'model_sha256': 'teacher'}, 'training': {'role': 'teacher', 'seed': 378}}}
        for name, role, alpha in (('hard', 'hard_only', 1), ('mixed', 'mixed_T5_a05', .5), ('soft', 'soft_only', 0)):
            training = copy.deepcopy(base)
            training['role'] = role
            training['configuration']['alpha'] = alpha
            models[name] = {'training': training}
        self.assertTrue(analysis.matched_controls(models, 'teacher')[0]['complete_three_objective_control'])
        for key, value in (('initial_weights_sha256', 'different'), ('teacher_file_sha256', 'wrong')):
            altered = copy.deepcopy(models)
            altered['soft']['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                analysis.matched_controls(altered, 'teacher')
        with self.assertRaises(ValueError):
            analysis.matched_controls(models, 'teacher', require_complete=True)


if __name__ == '__main__':
    unittest.main()
