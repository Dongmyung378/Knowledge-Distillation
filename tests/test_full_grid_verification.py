"""Small metadata/decision guards; no model execution or quality measurements."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from knowledge_distillation.evaluation import classification_metrics
from scripts import verify_full_grid_release as verifier


class FullGridVerificationGuards(unittest.TestCase):
    def selection_fixture(self):
        plan = verifier.read(Path(__file__).parents[1] / "configs/full-grid-plan.json")
        selected = "T5-a05"
        configuration = {**plan["student_training"], "alpha": .5, "temperature": 5.}
        selection = {"status": "completed", "selection_split": "validation",
                     "final_outcomes_used_for_selection": False, "controls_eligible_for_selection": False,
                     "selected_candidate_id": selected, "selected_configuration": configuration,
                     "deployment_seed": 378, "deployment_kind": "distilled",
                     "deployment": {"selected_on": "validation", "candidate_id": selected, "seed": 378},
                     "candidates": [{"id": i, "seeds": list(verifier.SEEDS)} for i in verifier.GRID_IDS]}
        training = {"plan": plan, "plan_sha256": verifier.canonical_sha256(plan), "deployment_seed": 378,
                    "selected_candidate_id": selected, "selected_configuration": configuration,
                    "deployment_model_name": "grid_T5-a05_seed_378", "final_outcomes_used_for_selection": False,
                    "final_reselection_allowed": False, "controls_eligible_for_selection": False}
        acceptance = {"selected_format": "FP32"}
        config = {"training_config": training, "selection": selection, "deployment_seed": 378,
                  "deployment_candidate_id": selected, "deployment_format": "FP32", "int8_acceptance": acceptance,
                  "deployment_selection_uses_reference_outcomes": False, "training_freeze_sha256": "training"}
        freeze = {"config": config, "models": {"deployment_student": {"bytes": 1, "sha256": "same"},
                                                  "grid_T5-a05_seed_378": {"bytes": 1, "sha256": "same"}}}
        summary = {"selection": copy.deepcopy(selection), "evaluation_does_not_select_models": True,
                   "training_freeze_sha256": "training", "deployment": {"format": "FP32",
                     "candidate_id": selected, "seed": 378, "int8_acceptance": acceptance}}
        return freeze, summary

    def test_exact_role_denominator_includes_all_controls_and_deployed_formats(self):
        roles = verifier.expected_roles()
        self.assertEqual(len(roles), 37)
        self.assertEqual(sum(r.startswith("grid_") for r in roles), 27)
        self.assertEqual(sum(r.startswith("control_") for r in roles), 3)
        self.assertIn("deployment_int8", roles)
        self.assertIn("baseline_seed_380", roles)

    def test_selection_report_change_control_selection_or_deployment_hash_is_rejected(self):
        freeze, summary = self.selection_fixture()
        self.assertEqual(verifier.checked_selection(freeze, summary)[1], "grid_T5-a05_seed_378")
        f, s = copy.deepcopy(freeze), copy.deepcopy(summary)
        f["config"]["deployment_format"] = f["config"]["int8_acceptance"]["selected_format"] = "INT8"
        s["deployment"]["format"] = "INT8"
        s["deployment"]["int8_acceptance"]["selected_format"] = "INT8"
        self.assertEqual(verifier.checked_selection(f, s)[1], "grid_T5-a05_seed_378")
        for kind in ("changed_summary", "control_selected", "changed_checkpoint", "holdout_selection"):
            f, s = copy.deepcopy(freeze), copy.deepcopy(summary)
            if kind == "changed_summary": s["selection"]["selected_candidate_id"] = "T3-a03"
            if kind == "control_selected": f["config"]["selection"]["selected_candidate_id"] = "T5-a00"
            if kind == "changed_checkpoint": f["models"]["deployment_student"]["sha256"] = "changed"
            if kind == "holdout_selection": f["config"]["deployment_selection_uses_reference_outcomes"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                verifier.checked_selection(f, s)

    def test_equal_aggregate_metrics_do_not_hide_changed_individual_decisions(self):
        reference = [{"id": "a", "label": 0, "prediction": 0, "confidence": .9},
                     {"id": "b", "label": 0, "prediction": 1, "confidence": .8}]
        actual = [{**reference[0], "prediction": 1}, {**reference[1], "prediction": 0}]
        metric = classification_metrics([0, 0], [0, 1])
        result = verifier.compare_outputs(metric, metric, actual, reference)
        self.assertTrue(result["metrics_match"])
        self.assertEqual(result["prediction_mismatches"], 2)
        self.assertEqual(result["status"], "failed_comparison")
        metrics_only = verifier.compare_outputs(metric, metric, actual)
        self.assertEqual(metrics_only["status"], "passed_metrics_only")
        self.assertFalse(metrics_only["prediction_comparison_available"])

    def test_saved_order_nonfinite_confidence_and_changed_confusion_are_rejected(self):
        records = [{"id": str(i), "label": i} for i in range(2)]
        rows = [{"id": str(i), "label": i, "prediction": i, "confidence": .9} for i in range(2)]
        metric = classification_metrics([0, 1], [0, 1])
        confusion = {"rows": "true_class_id", "columns": "predicted_class_id",
                     "class_names": [str(i) for i in range(43)], "matrix": metric["confusion_matrix"]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for kind in ("valid", "order", "nonfinite", "confusion"):
                candidate, matrix = copy.deepcopy(rows), copy.deepcopy(confusion)
                if kind == "order": candidate.reverse()
                if kind == "nonfinite": candidate[0]["confidence"] = float("nan")
                if kind == "confusion": matrix["matrix"][0][0] = 0
                (root / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in candidate), encoding="utf-8")
                (root / "confusion.json").write_text(json.dumps(matrix), encoding="utf-8")
                if kind == "valid":
                    self.assertEqual(verifier.checked_predictions(root, metric, records), rows)
                else:
                    with self.subTest(kind=kind), self.assertRaises(ValueError):
                        verifier.checked_predictions(root, metric, records)

    def test_confidence_rounding_is_reported_without_false_bit_identical_claim(self):
        row = {"id": "a", "label": 0, "prediction": 0, "confidence": .9}
        metric = classification_metrics([0], [0])
        result = verifier.compare_outputs(metric, metric, [{**row, "confidence": .9002}], [row])
        self.assertEqual(result["status"], "passed_saved_model_outcomes")
        self.assertAlmostEqual(result["maximum_top_class_confidence_difference"], .0002)
        with self.assertRaises(ValueError):
            verifier.compare_outputs(metric, metric, [row], [{**row, "id": "changed"}])


if __name__ == "__main__":
    unittest.main()
