from __future__ import annotations

import json
from pathlib import Path
import unittest


DATASET = Path(__file__).parents[1] / "evaluation_data/golden_questions.json"


class EvaluationDatasetTests(unittest.TestCase):
    def test_golden_dataset_has_answerable_and_unanswerable_cases(self) -> None:
        cases = json.loads(DATASET.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(cases), 20)
        self.assertTrue(any(case["answerable"] for case in cases))
        self.assertTrue(any(not case["answerable"] for case in cases))
        self.assertEqual(len({case["id"] for case in cases}), len(cases))


if __name__ == "__main__":
    unittest.main()
