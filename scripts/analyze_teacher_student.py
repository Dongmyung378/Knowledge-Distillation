"""Diagnose teacher/student differences using saved predictions only.

Example:
  python scripts/analyze_teacher_student.py --manifest reports/split-manifest.json \
    --freeze reports/evaluation-freeze.json --model teacher=EVALUATION/teacher \
    --training-report teacher=TRAINING/teachers/variant-0/training-report.json \
    --model mixed378=EVALUATION/distilled \
    --training-report mixed378=TRAINING/students/seed-378/T5-a05/training-report.json \
    --output .local/teacher-diagnostic

Repeat --model and --training-report for all seeds/candidates. Pure-soft controls
are identified by alpha=0, mixed controls by alpha=.5/T=5, and hard-only controls
by alpha=1 in hash-bound training reports. This imports no project/model package.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

CLASS_COUNT = 43
MATCHED_FIELDS = ('max_epochs', 'patience', 'batch_size', 'learning_rate',
                  'augmentation', 'deterministic')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def checked_manifest(manifest, *, expected_count=5850, expected_tracks=195):
    require(manifest.get('num_classes') == CLASS_COUNT and manifest.get('image_size') == 96,
            'Expected the declared 43-class RGB96 corpus')
    require(manifest.get('manifest_sha256') == canonical_sha({k: v for k, v in manifest.items() if k != 'manifest_sha256'}),
            'Manifest content hash differs')
    records = [row for row in manifest['records'] if row['split'] == 'locked_holdout']
    require(len(records) == expected_count and len({row['id'] for row in records}) == expected_count,
            'Final evaluation count/unique IDs differ')
    require(len({row['track_id'] for row in records}) == expected_tracks,
            'Physical-sign track denominator differs')
    require(all(type(row['label']) is int and 0 <= row['label'] < CLASS_COUNT for row in records),
            'Manifest labels differ from class IDs 0..42')
    tracks = defaultdict(set)
    for row in manifest['records']:
        tracks[row['track_id']].add(row['split'])
    require(all(len(splits) == 1 for splits in tracks.values()), 'A physical track crosses split boundaries')
    return records


def checked_freeze(freeze, manifest, source_root):
    require(freeze.get('freeze_sha256') == canonical_sha({k: v for k, v in freeze.items() if k != 'freeze_sha256'}),
            'Freeze checksum differs')
    require(freeze.get('manifest_sha256') == canonical_sha(manifest), 'Freeze binds a different manifest')
    expected = freeze['source_sha256']
    actual = {name: file_sha(source_root / name) for name in expected}
    require(actual == expected, 'Frozen executable/dependency source changed')
    return actual


def metrics_from_decisions(rows):
    matrix = [[0] * CLASS_COUNT for _ in range(CLASS_COUNT)]
    for row in rows:
        matrix[row['label']][row['prediction']] += 1
    f1s, recalls = [], []
    for label in range(CLASS_COUNT):
        support = sum(matrix[label])
        predicted = sum(row[label] for row in matrix)
        true_positive = matrix[label][label]
        f1s.append(2 * true_positive / (support + predicted) if support + predicted else 0.0)
        if support:
            recalls.append(true_positive / support)
    correct = sum(matrix[i][i] for i in range(CLASS_COUNT))
    return {'count': len(rows), 'correct': correct, 'accuracy': correct / len(rows),
            'macro_f1': statistics.mean(f1s), 'balanced_accuracy': statistics.mean(recalls)}, matrix


def checked_predictions(rows, metrics, records, manifest, freeze, *, expected_prediction_sha=None,
                        actual_prediction_sha=None, recorded_confusion=None):
    require(len(rows) == len(records), 'Saved predictions have a different denominator')
    require(metrics.get('split') == 'locked_holdout' and metrics.get('selection_permitted') is False,
            'Only frozen final predictions are accepted')
    require(metrics.get('manifest_sha256') == canonical_sha(manifest)
            and metrics.get('training_manifest_sha256') == canonical_sha(manifest), 'Prediction manifest binding differs')
    require(metrics.get('evaluation_records_sha256') == canonical_sha(records), 'Ordered evaluation record hash differs')
    require(metrics.get('freeze_sha256') == freeze['freeze_sha256'], 'Models use different evaluation freezes')
    require(metrics.get('class_names') == manifest.get('class_names'), 'Recorded class map differs')
    require(metrics.get('inventory_sha256') == manifest.get('inventory_sha256'), 'Recorded image inventory differs')
    require(metrics.get('model_sha256') in {item['sha256'] for item in freeze['models'].values()},
            'Prediction model is not in the supplied joint freeze')
    require(metrics.get('preprocessing', {}).get('image_size') == [96, 96]
            and metrics['preprocessing'].get('channels') == 'RGB'
            and metrics['preprocessing'].get('range') == [0, 1], 'Recorded preprocessing differs')
    if expected_prediction_sha is not None:
        require(expected_prediction_sha == actual_prediction_sha, 'Pinned saved-prediction SHA differs')
    for row, record in zip(rows, records, strict=True):
        require(row.get('id') == record['id'] and type(row.get('label')) is int and row.get('label') == record['label'],
                'Saved prediction ID/label/order differs')
        require(type(row.get('prediction')) is int and 0 <= row['prediction'] < CLASS_COUNT,
                'Saved prediction class is invalid')
        confidence = row.get('confidence')
        require(isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                and math.isfinite(confidence) and 0 <= confidence <= 1, 'Saved top-class confidence is invalid')
    observed, matrix = metrics_from_decisions(rows)
    for key, value in observed.items():
        require(key in metrics and abs(metrics[key] - value) <= 1e-12,
                'Stored metrics disagree with saved decisions: ' + key)
    if recorded_confusion is not None:
        require(recorded_confusion == matrix, 'Stored confusion matrix disagrees with decisions')
    return observed


def confidence_summary(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'at_least_0_9': sum(value >= .9 for value in values),
            'at_least_0_95': sum(value >= .95 for value in values),
            'at_least_0_99': sum(value >= .99 for value in values)}


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def track_bootstrap(track_rows, *, draws=1000, seed=378):
    require(type(draws) is int and 1 <= draws <= 5000, 'Bootstrap must use 1..5000 bounded draws')
    require(track_rows, 'A track-level comparison is required')
    generator = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [track_rows[generator.randrange(len(track_rows))] for _ in track_rows]
        denominator = sum(row['support'] for row in sampled)
        estimates.append(100 * sum(row['candidate_correct'] - row['reference_correct'] for row in sampled) / denominator)
    estimates.sort()
    return {'resampling_unit': 'whole observed physical-sign track, with replacement',
            'stratified_by_class': False, 'draws': draws, 'seed': seed,
            'track_count': len(track_rows), 'percentile_interval_95_pp': [percentile(estimates, .025), percentile(estimates, .975)],
            'scope': 'Descriptive cluster resampling of the already observed evaluation collection; not causal evidence, a teacher-seed interval, or untouched confirmatory inference.'}


def compare_decisions(reference, candidate, records, *, draws=1000, seed=378, baseline=None):
    states, classes, tracks = Counter(), defaultdict(Counter), defaultdict(Counter)
    confidences = defaultdict(list)
    samples = []
    for index, (r, c, record) in enumerate(zip(reference, candidate, records, strict=True)):
        label = record['label']
        a, b = r['prediction'] == label, c['prediction'] == label
        state = f'reference_{int(a)}_candidate_{int(b)}'
        states[state] += 1
        row = classes[label]
        row['support'] += 1
        row['reference_correct'] += a
        row['candidate_correct'] += b
        row['reference_wrong_candidate_correct'] += not a and b
        row['reference_correct_candidate_wrong'] += a and not b
        row['prediction_disagreement'] += r['prediction'] != c['prediction']
        row['same_wrong_prediction'] += not a and not b and r['prediction'] == c['prediction']
        track = tracks[record['track_id']]
        track['support'] += 1
        track['reference_correct'] += a
        track['candidate_correct'] += b
        if not a:
            confidences['all_reference_errors_top_probability'].append(r['confidence'])
        if not a and b:
            confidences['reference_wrong_candidate_correct_reference_top_probability'].append(r['confidence'])
            confidences['reference_wrong_candidate_correct_candidate_top_probability'].append(c['confidence'])
        if a and not b:
            confidences['reference_correct_candidate_wrong_reference_top_probability'].append(r['confidence'])
        sample = {'id': record['id'], 'label': label, 'track_id': record['track_id'],
                  'reference_prediction': r['prediction'], 'candidate_prediction': c['prediction'],
                  'reference_correct': int(a), 'candidate_correct': int(b),
                  'reference_top_probability': r['confidence'], 'candidate_top_probability': c['confidence']}
        if baseline is not None:
            h = baseline[index]['prediction'] == label
            states[f'teacher_{int(a)}_hard_only_{int(h)}_candidate_{int(b)}'] += 1
            sample['hard_only_prediction'] = baseline[index]['prediction']
            sample['hard_only_correct'] = int(h)
        samples.append(sample)
    class_rows = []
    for label in range(CLASS_COUNT):
        row = {'class_id': label, **classes[label]}
        for key in ('support', 'reference_correct', 'candidate_correct', 'reference_wrong_candidate_correct',
                    'reference_correct_candidate_wrong', 'prediction_disagreement', 'same_wrong_prediction'):
            row.setdefault(key, 0)
        row['candidate_minus_reference_correct'] = row['candidate_correct'] - row['reference_correct']
        row['candidate_minus_reference_recall_pp'] = 100 * row['candidate_minus_reference_correct'] / row['support'] if row['support'] else None
        class_rows.append(row)
    track_rows = [{'track_id': name, **values} for name, values in sorted(tracks.items())]
    repairs = states['reference_0_candidate_1']
    regressions = states['reference_1_candidate_0']
    reference_metrics, _ = metrics_from_decisions(reference)
    candidate_metrics, _ = metrics_from_decisions(candidate)
    return {'count': len(records), 'reference_wrong_candidate_correct': repairs,
            'reference_correct_candidate_wrong': regressions, 'net_correct': repairs - regressions,
            'accuracy_gain_pp': 100 * (repairs - regressions) / len(records),
            'reference_accuracy': reference_metrics['accuracy'], 'candidate_accuracy': candidate_metrics['accuracy'],
            'reference_macro_f1': reference_metrics['macro_f1'], 'candidate_macro_f1': candidate_metrics['macro_f1'],
            'macro_f1_gain': candidate_metrics['macro_f1'] - reference_metrics['macro_f1'],
            'prediction_disagreement': sum(row['prediction_disagreement'] for row in class_rows),
            'same_wrong_prediction': sum(row['same_wrong_prediction'] for row in class_rows),
            'correctness_states': dict(sorted(states.items())),
            'classes_candidate_higher_recall': sum(row['candidate_minus_reference_correct'] > 0 for row in class_rows),
            'classes_reference_higher_recall': sum(row['candidate_minus_reference_correct'] < 0 for row in class_rows),
            'classes_equal_recall': sum(row['candidate_minus_reference_correct'] == 0 for row in class_rows),
            'conditional_top_probability': {key: confidence_summary(values) for key, values in confidences.items()},
            'top_probability_is_not_true_class_probability': True,
            'track_cluster_bootstrap': track_bootstrap(track_rows, draws=draws, seed=seed)}, samples, class_rows, track_rows


def training_diagnostic(report_path, metric):
    report = read_json(report_path)
    require(report.get('model_sha256') == metric['model_sha256'], 'Training report binds a different model')
    require(report.get('parameter_count') == metric.get('model_parameters'), 'Training/evaluation architecture size differs')
    require(report.get('selection_split') == 'validation' and report.get('locked_holdout_evaluated') is False,
            'Training/checkpoint selection used a different split')
    config = report['configuration']
    if report['kind'] == 'teacher':
        role = 'teacher'
        stage = next(row for row in report['stages'] if row['stage'] == report['selected_stage'])
        history_path = report_path.parent / (stage['stage'] + '-history.json')
        best_epoch, epochs = stage['best_epoch'], stage['epochs_run']
    else:
        alpha, temperature = config.get('alpha'), config.get('temperature')
        require(isinstance(alpha, (int, float)) and 0 <= alpha <= 1 and temperature > 0, 'Invalid loss recipe')
        role = 'hard_only' if alpha == 1 else 'soft_only' if alpha == 0 and temperature == 5 else 'mixed_T5_a05' if alpha == .5 and temperature == 5 else 'grid_candidate'
        history_path = report_path.parent / 'history.json'
        best_epoch, epochs = report['best_epoch'], report['epochs_run']
    history = read_json(history_path)
    require(len(history['val_accuracy']) == epochs and best_epoch == 1 + max(range(epochs), key=lambda i: history['val_accuracy'][i]),
            'Selected checkpoint epoch disagrees with history')
    require(abs(history['val_accuracy'][best_epoch - 1] - report['validation_accuracy']) <= 1e-6,
            'Reloaded selected validation score disagrees with history')
    require(canonical_sha(config) == report['configuration_sha256'], 'Training recipe hash differs')
    selected_train = history['accuracy'][best_epoch - 1]
    return {'role': role, 'seed': report['seed'], 'configuration': config, 'epochs_run': epochs, 'best_epoch': best_epoch,
            'selected_train_accuracy': selected_train, 'selected_validation_accuracy': report['validation_accuracy'],
            'selected_validation_ce': report['validation_ce'],
            'train_minus_validation_accuracy_pp': 100 * (selected_train - report['validation_accuracy']),
            'accuracy_conditions': {'train': 'Recorded online training with random augmentation and train-mode dropout/normalization.',
                                    'validation': 'Clean ordered images, training=False and unweighted accuracy. These are different prediction conditions.'},
            'learning_rate_history': history['learning_rate'], 'initial_weights_sha256': report.get('initial_weights_sha256'),
            'initial_weights_file_sha256': report.get('initial_weights_file_sha256'),
            'teacher_file_sha256': report.get('teacher_file_sha256'), 'teacher_immutable': report.get('teacher_immutable'),
            'source_sha256': report['source_sha256'], 'model_sha256': report['model_sha256'],
            'report_sha256': file_sha(report_path), 'history_sha256': file_sha(history_path)}


def matched_controls(models, teacher_name, *, require_complete=False):
    roles = defaultdict(dict)
    for name, model in models.items():
        training = model['training']
        if training['role'] in ('hard_only', 'mixed_T5_a05', 'soft_only'):
            seed = training['seed']
            require(training['role'] not in roles[seed], 'Duplicate role/seed in matched controls')
            roles[seed][training['role']] = name
    if require_complete:
        require(set(roles) == {378, 379, 380} and all(set(group) == {'hard_only', 'mixed_T5_a05', 'soft_only'} for group in roles.values()),
                'Three-seed hard/mixed/pure-soft controls are incomplete')
    results = []
    teacher_sha = models[teacher_name]['metrics']['model_sha256']
    for seed, names in sorted(roles.items()):
        entries = [models[name]['training'] for name in names.values()]
        reference = entries[0]
        require(reference['initial_weights_sha256'], 'Missing shared initial weights')
        require(all(item['initial_weights_sha256'] == reference['initial_weights_sha256']
                    and item['initial_weights_file_sha256'] == reference['initial_weights_file_sha256']
                    and all(item['configuration'][key] == reference['configuration'][key] for key in MATCHED_FIELDS)
                    for item in entries), 'Hard/mixed/pure-soft initialization or budget policy differs')
        for item in entries:
            if item['role'] != 'hard_only':
                require(item['teacher_file_sha256'] == teacher_sha and item['teacher_immutable'] is True,
                        'Mixed/pure-soft controls used a different or mutable teacher')
        results.append({'seed': seed, 'models': names, 'complete_three_objective_control': len(names) == 3,
                        'shared_initialization_and_budget_rules': True, 'fixed_teacher_sha256': teacher_sha,
                        'same_realized_epochs': len({item['epochs_run'] for item in entries}) == 1,
                        'same_actual_learning_rate_history': all(item['learning_rate_history'] == reference['learning_rate_history'] for item in entries),
                        'alpha_hard_label_weight': {role: models[name]['training']['configuration']['alpha'] for role, name in names.items()},
                        'soft_temperature': {role: models[name]['training']['configuration']['temperature'] for role, name in names.items()},
                        'interpretation_limit': 'Loss-dependent learning-rate scheduling and validation checkpoint selection are part of the observed pipeline. This is not a pure-gradient causal decomposition.'})
    return results


def assignments(values):
    result = {}
    for value in values:
        name, separator, location = value.partition('=')
        require(separator and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) and name not in result,
                'Use unique safe NAME=PATH assignments')
        result[name] = Path(location)
    return result


def save_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--training-report', action='append', required=True)
    parser.add_argument('--teacher-name', default='teacher')
    parser.add_argument('--prediction-references', type=Path)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).parents[1])
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=378)
    parser.add_argument('--require-three-seed-controls', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve earlier diagnostics; use a new output directory')
    require(1 <= args.bootstrap_draws <= 5000, 'Bootstrap draws must be bounded to 1..5000')
    manifest, freeze = read_json(args.manifest), read_json(args.freeze)
    records = checked_manifest(manifest)
    sources_before = checked_freeze(freeze, manifest, args.source_root)
    directories, reports = assignments(args.model), assignments(args.training_report)
    require(set(directories) == set(reports) and args.teacher_name in directories, 'Every model needs its hash-bound training report')
    references = read_json(args.prediction_references)['predictions'] if args.prediction_references else []
    models = {}
    for name, directory in directories.items():
        prediction_path, metric_path = directory / 'predictions.jsonl', directory / 'metrics.json'
        rows = [json.loads(line) for line in prediction_path.read_text(encoding='utf-8').splitlines() if line]
        metric = read_json(metric_path)
        # References use producer directory names, independent of display aliases.
        pinned = [entry for entry in references if entry['scope'] == 'locked_holdout' and entry['model'] == directory.name]
        if references:
            require(len(pinned) == 1, 'Pinned prediction reference is missing/ambiguous')
        prediction_sha = file_sha(prediction_path)
        confusion_path = directory / 'confusion.json'
        derived = checked_predictions(rows, metric, records, manifest, freeze,
                                      expected_prediction_sha=pinned[0]['sha256'] if pinned else metric.get('predictions_sha256'),
                                      actual_prediction_sha=prediction_sha,
                                      recorded_confusion=read_json(confusion_path)['matrix'] if confusion_path.exists() else None)
        training = training_diagnostic(reports[name], metric)
        require(training['source_sha256'] == sources_before, 'Training and frozen source differ')
        models[name] = {'rows': rows, 'metrics': metric, 'derived_metrics': derived, 'training': training,
                        'predictions_sha256': prediction_sha, 'metrics_sha256': file_sha(metric_path),
                        'prediction_file_hash_externally_pinned': bool(pinned) or bool(metric.get('predictions_sha256'))}
    require(models[args.teacher_name]['training']['role'] == 'teacher', 'Declared teacher is not a teacher training artifact')
    teacher_sha = models[args.teacher_name]['metrics']['model_sha256']
    for name, model in models.items():
        training = model['training']
        if training['role'] not in ('teacher', 'hard_only'):
            require(training['teacher_file_sha256'] == teacher_sha and training['teacher_immutable'] is True,
                    'A candidate/control used a different or mutable teacher: ' + name)
    controls = matched_controls(models, args.teacher_name, require_complete=args.require_three_seed_controls)
    comparisons, tables = {}, {}
    hard_by_seed = {row['seed']: row['models'].get('hard_only') for row in controls}
    for name, model in models.items():
        if name == args.teacher_name:
            continue
        hard_name = hard_by_seed.get(model['training']['seed'])
        result, samples, classes, tracks = compare_decisions(models[args.teacher_name]['rows'], model['rows'], records,
                                                            draws=args.bootstrap_draws, seed=args.bootstrap_seed,
                                                            baseline=models[hard_name]['rows'] if hard_name else None)
        result.update(reference_model=args.teacher_name, candidate_model=name,
                      role=model['training']['role'], training_seed=model['training']['seed'])
        comparisons[name + '-vs-teacher'] = result
        tables[name + '-vs-teacher'] = (samples, classes, tracks)
    control_comparisons = {}
    for group in controls:
        names = group['models']
        for candidate_role, reference_role in (('mixed_T5_a05', 'hard_only'), ('mixed_T5_a05', 'soft_only'), ('soft_only', 'hard_only')):
            if candidate_role not in names or reference_role not in names:
                continue
            candidate, reference = names[candidate_role], names[reference_role]
            result, _, _, _ = compare_decisions(models[reference]['rows'], models[candidate]['rows'], records,
                                                draws=args.bootstrap_draws, seed=args.bootstrap_seed)
            result.update(reference_model=reference, candidate_model=candidate, training_seed=group['seed'],
                          contrast=candidate_role + '-vs-' + reference_role)
            control_comparisons[f'{candidate}-vs-{reference}'] = result
    control_means = {}
    for contrast in sorted({row['contrast'] for row in control_comparisons.values()}):
        rows = [row for row in control_comparisons.values() if row['contrast'] == contrast]
        gains = [row['accuracy_gain_pp'] for row in rows]
        control_means[contrast] = {'training_seed_count': len(rows), 'training_seeds': sorted(row['training_seed'] for row in rows),
                                   'mean_accuracy_gain_pp': statistics.mean(gains),
                                   'sample_std_accuracy_gain_pp': statistics.stdev(gains) if len(rows) > 1 else None,
                                   'mean_macro_f1_gain': statistics.mean(row['macro_f1_gain'] for row in rows),
                                   'scope': 'Descriptive mean over completed paired student seeds; not a confidence interval or causal attribution.'}
    require(checked_freeze(freeze, manifest, args.source_root) == sources_before, 'Frozen source changed during analysis')
    summary = {'schema_version': 1, 'status': 'completed_saved_prediction_diagnostic',
               'completed_utc': datetime.now(timezone.utc).isoformat(), 'split': 'locked_holdout',
               'count': len(records), 'physical_track_count': len({row['track_id'] for row in records}),
               'freeze_sha256': freeze['freeze_sha256'], 'manifest_file_sha256': file_sha(args.manifest),
               'evaluation_records_sha256': canonical_sha(records), 'source_sha256': sources_before,
               'analyzer_sha256': file_sha(__file__),
               'analysis_parameters': {'bootstrap_draws': args.bootstrap_draws, 'bootstrap_seed': args.bootstrap_seed,
                                       'mixed_control_temperature': 5, 'mixed_control_alpha': .5, 'soft_only_control_alpha': 0},
               'model_execution_performed': False, 'training_or_calibration_performed': False,
               'model_or_configuration_reselected': False, 'final_evaluation_previously_observed': True,
               'models': {name: {key: value for key, value in model.items() if key not in ('rows', 'metrics')}
                          for name, model in models.items()},
               'teacher_comparisons': comparisons, 'matched_controls': controls,
               'mixed_soft_hard_comparisons': control_comparisons,
               'mixed_soft_hard_seed_means': control_means,
               'causal_interpretation_limits': [
                   'Observed repairs/regressions establish error differences; they do not prove that distillation itself caused every repair.',
                   'A shared teacher over three student seeds does not measure teacher-training-seed variance.',
                   'Recorded train accuracy uses augmentation/train-mode dropout/normalization; clean validation uses different conditions.',
                   'Loss-dependent ReduceLROnPlateau, realized stopping epochs and validation-selected checkpoints can differ.',
                   'Only predicted-class maximum probabilities were saved; true-class probability, KL, entropy and per-sample training gradients cannot be reconstructed.',
                   'The evaluation has already been observed. Cluster intervals are descriptive resampling, not untouched confirmation or production-reliability claims.',
               ]}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, (samples, classes, tracks) in tables.items():
        save_csv(args.output / (name + '-samples.csv'), samples)
        save_csv(args.output / (name + '-classes.csv'), classes)
        save_csv(args.output / (name + '-tracks.csv'), tracks)
    with (args.output / 'summary.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': summary['status'], 'models': len(models), 'count': len(records),
                      'tracks': summary['physical_track_count'], 'matched_control_sets': len(controls),
                      'all_control_sets_complete': all(group['complete_three_objective_control'] for group in controls),
                      'summary_sha256': file_sha(args.output / 'summary.json')}))


if __name__ == '__main__':
    main()
