"""Plot completed, frozen saved-prediction diagnostics without model execution.

Usage:
  python scripts/plot_teacher_student_analysis.py --summary ANALYSIS/summary.json \
    --selected-id T5-a05 --output NEW_FIGURE_DIRECTORY

The caller supplies the validation-selected ID. This tool never ranks models
using final outcomes. All three hard-only, T5-mixed and T5-pure-soft control
groups must exist before any figure is created. Synthetic fixtures cannot be
rendered by the CLI. Matplotlib is imported only after all data guards pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SEEDS = (378, 379, 380)
COLORS = {'teacher': '#694483', 'selected': '#88994A',
          'hard_only': '#939099', 'mixed_T5_a05': '#88994A', 'soft_only': '#7C828C'}
SHA_PATTERN = re.compile(r'[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, lower=None, upper=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), label + ': finite numeric value required')
    require(lower is None or value >= lower, label + ': value below permitted range')
    require(upper is None or value <= upper, label + ': value above permitted range')
    return float(value)


def recipe_from_id(selected_id):
    match = re.fullmatch(r'T(3|5|10)-a(01|03|05)', selected_id)
    require(match is not None, 'Supply one of the nine declared KD IDs; no recipe may be guessed')
    return int(match.group(1)), int(match.group(2)) / 10


def prepare_plot_data(summary, selected_id, *, allow_synthetic_fixture=False):
    require(not summary.get('_synthetic_fixture') or allow_synthetic_fixture,
            'Synthetic fixtures are not project results and cannot be rendered')
    require(summary.get('schema_version') == 1 and summary.get('status') == 'completed_saved_prediction_diagnostic',
            'A completed saved-prediction diagnostic is required')
    require(summary.get('split') == 'locked_holdout' and summary.get('count') == 5850
            and summary.get('physical_track_count') == 195, 'Expected the same 5,850 images/195 physical tracks')
    for field in ('freeze_sha256', 'manifest_file_sha256', 'evaluation_records_sha256', 'analyzer_sha256'):
        require(isinstance(summary.get(field), str) and SHA_PATTERN.fullmatch(summary[field]), 'Missing actual binding: ' + field)
    for field in ('model_execution_performed', 'training_or_calibration_performed', 'model_or_configuration_reselected'):
        require(summary.get(field) is False, 'Diagnostic scope must not execute/refit/reselect models')
    require(summary.get('final_evaluation_previously_observed') is True, 'Previously exposed evaluation scope must be explicit')
    if 'selected_candidate_id' in summary:
        require(summary['selected_candidate_id'] == selected_id, 'Supplied ID differs from the recorded validation selection')
    temperature, alpha = recipe_from_id(selected_id)
    models = summary.get('models', {})
    require(isinstance(models, dict) and models, 'Evaluated model objects are missing')
    for name, model in models.items():
        require(isinstance(model, dict) and isinstance(model.get('training'), dict), 'Missing model training binding: ' + name)
        for field in ('predictions_sha256', 'metrics_sha256'):
            require(isinstance(model.get(field), str) and SHA_PATTERN.fullmatch(model[field]), 'Missing saved-output hash: ' + name)
        require(isinstance(model['training'].get('model_sha256'), str)
                and SHA_PATTERN.fullmatch(model['training']['model_sha256']), 'Missing frozen model hash: ' + name)
        metric = model.get('derived_metrics', {})
        correct = finite(metric.get('correct'), name + '.correct', 0, summary['count'])
        require(correct.is_integer() and metric.get('count') == summary['count'], 'Model denominator/correct count differs: ' + name)
        accuracy = finite(metric.get('accuracy'), name + '.accuracy', 0, 1)
        finite(metric.get('macro_f1'), name + '.macro_f1', 0, 1)
        require(abs(accuracy - correct / summary['count']) <= 1e-12, 'Accuracy disagrees with evaluated count: ' + name)
    teachers = [name for name, model in models.items() if model['training'].get('role') == 'teacher']
    require(len(teachers) == 1, 'Exactly one fixed teacher must be recorded')
    teacher_name = teachers[0]
    teacher = models[teacher_name]
    teacher_sha = teacher['training']['model_sha256']
    teacher_correct = teacher['derived_metrics']['correct']
    selected = []
    for seed in SEEDS:
        candidates = []
        for name, model in models.items():
            training, config = model['training'], model['training'].get('configuration', {})
            if training.get('seed') == seed and config.get('temperature') == temperature and config.get('alpha') == alpha:
                candidates.append(name)
        require(len(candidates) == 1, 'Selected recipe/seed is missing or ambiguous: ' + str(seed))
        name = candidates[0]
        training = models[name]['training']
        require(training.get('teacher_file_sha256') == teacher_sha and training.get('teacher_immutable') is True,
                'Selected candidates do not share the fixed teacher')
        entries = [row for row in summary.get('teacher_comparisons', {}).values()
                   if row.get('candidate_model') == name and row.get('reference_model') == teacher_name]
        require(len(entries) == 1, 'Actual paired teacher comparison is missing/ambiguous: ' + name)
        paired = entries[0]
        require(paired.get('count') == summary['count'] and paired.get('training_seed') == seed,
                'Selected paired comparison scope differs')
        repairs = finite(paired.get('reference_wrong_candidate_correct'), 'repairs', 0, summary['count'])
        regressions = finite(paired.get('reference_correct_candidate_wrong'), 'regressions', 0, summary['count'])
        require(repairs.is_integer() and regressions.is_integer(), 'Paired error counts must be integers')
        net = int(repairs - regressions)
        require(paired.get('net_correct') == net
                and net == models[name]['derived_metrics']['correct'] - teacher_correct,
                'Paired error decomposition disagrees with saved model counts')
        require(abs(finite(paired.get('accuracy_gain_pp'), 'paired gain') - 100 * net / summary['count']) <= 1e-10,
                'Paired gain disagrees with error decomposition')
        selected.append({'seed': seed, 'model': name, 'repairs': int(repairs),
                         'regressions': int(regressions), 'net_correct': net,
                         'accuracy_gain_pp': 100 * net / summary['count']})
    groups = summary.get('matched_controls', [])
    require(len(groups) == 3 and {row.get('seed') for row in groups} == set(SEEDS),
            'All three matched control groups are required')
    by_seed = {}
    for group in groups:
        require(group.get('complete_three_objective_control') is True
                and group.get('shared_initialization_and_budget_rules') is True
                and group.get('fixed_teacher_sha256') == teacher_sha,
                'Hard/mixed/pure-soft matched control verification is incomplete')
        names = group.get('models', {})
        require(set(names) == {'hard_only', 'mixed_T5_a05', 'soft_only'}, 'A control objective is missing')
        require(all(name in models for name in names.values()), 'Control group references a missing evaluated model')
        for role, name in names.items():
            training, config = models[name]['training'], models[name]['training']['configuration']
            expected_alpha = {'hard_only': 1, 'mixed_T5_a05': .5, 'soft_only': 0}[role]
            require(training.get('role') == role and training.get('seed') == group['seed']
                    and config.get('alpha') == expected_alpha, 'Control role/seed/objective disagrees with training recipe')
            if role != 'hard_only':
                require(config.get('temperature') == 5 and training.get('teacher_file_sha256') == teacher_sha
                        and training.get('teacher_immutable') is True, 'Soft controls must use T5 and the fixed teacher')
        by_seed[group['seed']] = names
    control_stats = []
    for role in ('hard_only', 'soft_only', 'mixed_T5_a05'):
        runs = [{'seed': seed, 'model': by_seed[seed][role],
                 'accuracy_percent': 100 * models[by_seed[seed][role]]['derived_metrics']['accuracy']}
                for seed in SEEDS]
        values = [row['accuracy_percent'] for row in runs]
        control_stats.append({'role': role, 'runs': runs, 'mean_accuracy_percent': statistics.mean(values),
                              'sample_sd_accuracy_pp': statistics.stdev(values)})
    return {'selected_id': selected_id, 'selected_temperature': temperature, 'selected_alpha': alpha,
            'selection_origin': 'Caller-provided validation-selected ID; never inferred from final accuracy.',
            'count': summary['count'], 'physical_tracks': summary['physical_track_count'],
            'fixed_teacher_model': teacher_name, 'fixed_teacher_accuracy_percent': 100 * teacher['derived_metrics']['accuracy'],
            'fixed_teacher_model_sha256': teacher_sha, 'selected_candidate_pairs': selected,
            'T5_controls': control_stats, 'errorbar_policy': 'Sample SD over three student-training seeds; not confidence intervals.',
            'alpha0_caveat': 'No label CE in the training objective; validation labels still choose checkpoints and learning-rate changes.',
            'alpha_weight_caveat': 'Changing alpha also rescales KL; at T5 mixed uses 12.5*KL+.5*CE and pure-soft uses25*KL.',
            'scope': 'Previously observed evaluation collection; descriptive paired-pipeline comparison, not causal identification.'}


def make_plot(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    png, svg = output / 'teacher-student-analysis.png', output / 'teacher-student-analysis.svg'
    require(not png.exists() and not svg.exists() and not (output / 'teacher-student-analysis-plot.json').exists(),
            'Preserve previous diagnostic figures')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                         'axes.labelcolor': '#403846', 'text.color': '#403846',
                         'figure.facecolor': '#FBFAFC', 'axes.facecolor': 'white',
                         'savefig.facecolor': '#FBFAFC', 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(1, 2, figsize=(15.8, 7.2))
    figure.subplots_adjust(left=.07, right=.975, top=.785, bottom=.32, wspace=.28)
    x = np.arange(3)
    repairs = [row['repairs'] for row in data['selected_candidate_pairs']]
    regressions = [row['regressions'] for row in data['selected_candidate_pairs']]
    axes[0].bar(x - .18, repairs, .36, color=COLORS['selected'], label='Teacher wrong, selected KD correct')
    axes[0].bar(x + .18, regressions, .36, color=COLORS['teacher'], label='Teacher correct, selected KD wrong')
    for index, row in enumerate(data['selected_candidate_pairs']):
        for offset, key in ((-.18, 'repairs'), (.18, 'regressions')):
            axes[0].annotate(f"{row[key]:,}", (index + offset, row[key]), xytext=(0, 5), textcoords='offset points',
                             ha='center', fontsize=10, fontweight='semibold')
        axes[0].text(index, max(repairs + regressions) * 1.12,
                     f"Net {row['net_correct']:+,}\n{row['accuracy_gain_pp']:+.2f} pp", ha='center', fontsize=9.5)
    axes[0].set_xticks(x, [str(seed) for seed in SEEDS])
    axes[0].set_xlabel('Student training seed')
    axes[0].set_ylabel('Paired image count')
    axes[0].set_ylim(0, max(1, max(repairs + regressions)) * 1.4)
    axes[0].set_title(f"Selected {data['selected_id']}: paired teacher errors", loc='left', pad=16,
                      color=COLORS['teacher'], fontsize=12.5, weight='bold')
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -.16), fontsize=9, frameon=False)
    names = ['Hard labels\nalpha=1', 'Pure soft T5\nalpha=0', 'Mixed T5\nalpha=0.5']
    means = [row['mean_accuracy_percent'] for row in data['T5_controls']]
    deviations = [row['sample_sd_accuracy_pp'] for row in data['T5_controls']]
    markers = ('o', 'D', '^')
    for index, row in enumerate(data['T5_controls']):
        axes[1].bar(index, means[index], .6, color=COLORS[row['role']], alpha=.9)
        axes[1].errorbar(index, means[index], yerr=deviations[index], color='#403846', capsize=5,
                         linewidth=1.3, fmt='none', zorder=5)
        for offset, run, marker in zip((-.13, 0, .13), row['runs'], markers, strict=True):
            axes[1].scatter(index + offset, run['accuracy_percent'], marker=marker, s=34,
                            facecolor='white', edgecolor='#403846', linewidth=1, zorder=6)
        axes[1].text(index, means[index] + deviations[index] + 2.5,
                     f'{means[index]:.2f}%\nSD {deviations[index]:.2f} pp', ha='center', fontsize=9.5)
    teacher_accuracy = data['fixed_teacher_accuracy_percent']
    axes[1].axhline(teacher_accuracy, color=COLORS['teacher'], linestyle='--', linewidth=1.8,
                    label=f'One fixed teacher: {teacher_accuracy:.2f}%')
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, max(105, max(mean + sd for mean, sd in zip(means, deviations)) + 8))
    axes[1].set_yticks([0, 20, 40, 60, 80, 100])
    axes[1].set_title('T5 objective controls: three student seeds', loc='left', pad=16,
                      color=COLORS['teacher'], fontsize=12.5, weight='bold')
    handles = [plt.Line2D([0], [0], color=COLORS['teacher'], linestyle='--', label=f'Fixed teacher {teacher_accuracy:.2f}%')]
    handles.extend(plt.Line2D([0], [0], linestyle='none', marker=marker, markerfacecolor='white',
                            markeredgecolor='#403846', label=str(seed)) for marker, seed in zip(markers, SEEDS))
    axes[1].legend(handles=handles, loc='upper left', bbox_to_anchor=(0, -.16), ncols=2, fontsize=9, frameon=False)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['left', 'bottom']].set_color('#B6ADBF')
        axis.tick_params(colors='#554B5F', labelsize=10)
        axis.set_axisbelow(True)
        axis.grid(axis='y', color='#EAE5EE', linewidth=.75)
    figure.suptitle('Teacher and student: observed error differences', x=.07, y=.97, ha='left',
                    fontsize=19, fontweight='bold', color=COLORS['teacher'])
    figure.text(.07, .915, f"{data['count']:,} identical images | {data['physical_tracks']} physical-sign tracks | previously reported evaluation set",
                ha='left', fontsize=11, color='#665D70')
    notes = [
        'The selected ID was supplied from validation selection; this figure does not rank models using final outcomes.',
        'Means and error bars: three student seeds and sample SD. The teacher is one shared checkpoint, not three teacher-seed runs.',
        'Pure-soft alpha=0 removes label CE from training; validation labels still select checkpoints and the learning-rate schedule.',
        'Alpha changes both CE and KL weights. Different realized LR paths/epochs limit causal attribution of these pipeline comparisons.',
    ]
    for y, note in zip((.15, .115, .08, .045), notes, strict=True):
        figure.text(.07, y, note, ha='left', fontsize=9.3, color='#665D70')
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=180, bbox_inches='tight', pad_inches=.2)
    figure.savefig(svg, bbox_inches='tight', pad_inches=.2)
    plt.close(figure)
    return png, svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--selected-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.summary.read_bytes()
        summary = json.loads(raw)
        data = prepare_plot_data(summary, args.selected_id)
        png, svg = make_plot(data, args.output)
        result = {'schema_version': 1, 'display_only': True, 'source_summary_sha256': hashlib.sha256(raw).hexdigest(),
                  'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'freeze_sha256': summary['freeze_sha256'], 'data': data,
                  'files': {path.name: {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                            for path in (png, svg)}}
        with (args.output / 'teacher-student-analysis-plot.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({'display_only': True, 'selected_id': args.selected_id, 'paired_seeds': list(SEEDS),
                          'all_controls_complete': True, 'models_executed': False}))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(json.dumps({'success': False, 'error': str(error), 'models_executed': False}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
