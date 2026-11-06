import unittest

import numpy as np
import torch

from evaluate_teacher_ensemble import compare


class EnsembleTests(unittest.TestCase):
    def test_subset_count_and_identical_members(self):
        logits = np.array([[[2, 0], [0, 2]], [[2, 0], [0, 2]]])
        result = compare(logits, np.array([0, 1]), ["a", "b"])
        self.assertEqual(len(result["subsets"]), 3)
        self.assertTrue(all(row["correct"] == 2 for row in result["subsets"]))
        self.assertEqual(result["pairs"][0]["disagreements"], 0)

    def test_probability_average_and_disagreement(self):
        logits = torch.tensor([[[8., 0.], [1., 2.]], [[0., 1.], [3., 0.]]])
        labels = torch.tensor([0, 0])
        result = compare(logits, labels, ["a", "b"], 4)
        pair = next(row for row in result["subsets"] if len(row["members"]) == 2)
        expected = int(((logits / 4).softmax(-1).mean(0).argmax(1) == labels).sum())
        self.assertEqual(pair["correct"], expected)
        self.assertEqual(result["pairs"][0]["disagreements"], 2)
        self.assertEqual(result["pairs"][0]["a_only_correct"], 1)
        self.assertEqual(result["pairs"][0]["b_only_correct"], 1)

    def test_reject_misalignment_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            compare(np.zeros((2, 4, 10)), np.zeros(3), ["a", "b"])
        with self.assertRaises(ValueError):
            compare(np.full((1, 2, 10), np.nan), np.zeros(2), ["a"])
