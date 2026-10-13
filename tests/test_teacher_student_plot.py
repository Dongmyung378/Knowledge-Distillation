"""Plot-data guards on synthetic scalars only; no figures or models generated."""
import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location('teacher_plot', Path(__file__).parents[1] / 'scripts/plot_teacher_student_analysis.py')
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def fixture():
    digest = 'a' * 64
    summary = {'_synthetic_fixture': True, 'schema_version': 1, 'status': 'completed_saved_prediction_diagnostic',
               'split': 'locked_holdout', 'count': 5850, 'physical_track_count': 195,
               'freeze_sha256': digest, 'manifest_file_sha256': digest,
               'evaluation_records_sha256': digest, 'analyzer_sha256': digest,
               'model_execution_performed': False, 'training_or_calibration_performed': False,
               'model_or_configuration_reselected': False, 'final_evaluation_previously_observed': True,
               'models': {}, 'teacher_comparisons': {}, 'matched_controls': []}
    def model(role, seed, alpha, correct):
        return {'predictions_sha256': digest, 'metrics_sha256': digest,
                'derived_metrics': {'count': 5850, 'correct': correct, 'accuracy': correct / 5850, 'macro_f1': .5},
                'training': {'role': role, 'seed': seed, 'model_sha256': digest,
                             'teacher_file_sha256': digest, 'teacher_immutable': True,
                             'configuration': {'temperature': 5, 'alpha': alpha}}}
    summary['models']['teacher'] = model('teacher', 378, None, 4000)
    for index, seed in enumerate(plot.SEEDS):
        names = {'hard_only': f'hard{seed}', 'mixed_T5_a05': f'mixed{seed}', 'soft_only': f'soft{seed}'}
        for role, alpha, base in (('hard_only', 1, 3900), ('mixed_T5_a05', .5, 4200), ('soft_only', 0, 3800)):
            summary['models'][names[role]] = model(role, seed, alpha, base + index * 100)
        net = 200 + index * 100
        summary['teacher_comparisons'][f'mixed{seed}-vs-teacher'] = {
            'candidate_model': f'mixed{seed}', 'reference_model': 'teacher', 'count': 5850,
            'training_seed': seed, 'reference_wrong_candidate_correct': net + 100,
            'reference_correct_candidate_wrong': 100, 'net_correct': net,
            'accuracy_gain_pp': 100 * net / 5850}
        summary['matched_controls'].append({'seed': seed, 'models': names,
                                           'complete_three_objective_control': True,
                                           'shared_initialization_and_budget_rules': True,
                                           'fixed_teacher_sha256': digest})
    return summary


class TeacherPlotDataTests(unittest.TestCase):
    def test_exact_config_lookup_mean_and_sample_sd(self):
        data = plot.prepare_plot_data(fixture(), 'T5-a05', allow_synthetic_fixture=True)
        self.assertEqual([row['net_correct'] for row in data['selected_candidate_pairs']], [200, 300, 400])
        mixed = next(row for row in data['T5_controls'] if row['role'] == 'mixed_T5_a05')
        self.assertAlmostEqual(mixed['mean_accuracy_percent'], 4300 / 5850 * 100)
        self.assertAlmostEqual(mixed['sample_sd_accuracy_pp'], 100 / 5850 * 100)

    def test_synthetic_fixture_cannot_be_rendered_by_default(self):
        with self.assertRaises(ValueError):
            plot.prepare_plot_data(fixture(), 'T5-a05')

    def test_incomplete_pure_soft_or_selected_seed_is_rejected(self):
        for mode in ('soft', 'selected'):
            summary = fixture()
            if mode == 'soft': summary['matched_controls'][0]['models'].pop('soft_only')
            else: summary['models'].pop('mixed380')
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                plot.prepare_plot_data(summary, 'T5-a05', allow_synthetic_fixture=True)

    def test_inconsistent_error_counts_or_unknown_recipe_is_rejected(self):
        summary = fixture()
        summary['teacher_comparisons']['mixed378-vs-teacher']['net_correct'] += 1
        with self.assertRaises(ValueError):
            plot.prepare_plot_data(summary, 'T5-a05', allow_synthetic_fixture=True)
        with self.assertRaises(ValueError):
            plot.prepare_plot_data(fixture(), 'T7-a02', allow_synthetic_fixture=True)


if __name__ == '__main__':
    unittest.main()
