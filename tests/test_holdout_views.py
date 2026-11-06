import unittest

import numpy as np
import torch
import torch.nn.functional as F

from compare_holdout_views import batch_predictions, offsets, pair_statistics, translated
from train import predict_logits


class ViewsTest(unittest.TestCase):
    def test_reflect_crop_and_affine_normalization_commute(self):
        x = torch.arange(75.).reshape(1, 3, 5, 5)
        mean, std = torch.tensor([1., 2., 3.])[None, :, None, None], 2.0
        for radius in (1, 2):
            for dy, dx in offsets(radius):
                y = translated(x, dy, dx)
                padded = F.pad(x, (2, 2, 2, 2), mode="reflect")
                torch.testing.assert_close(y, padded[:, :, 2 + dy:7 + dy, 2 + dx:7 + dx])
                torch.testing.assert_close(translated((x - mean) / std, dy, dx), (y - mean) / std)

    def test_baseline_matches_existing_predict(self):
        torch.manual_seed(1)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 5 * 5, 10)).eval()
        x = torch.randn(4, 3, 5, 5)
        result = batch_predictions(model, x)
        torch.testing.assert_close(result["flip"], predict_logits(model, x, "flip").argmax(1))
        self.assertEqual(set(result), {"flip", "translate1_flip", "translate2_flip"})
        flip_only = batch_predictions(model, x, flip_only=True)
        self.assertEqual(set(flip_only), {"flip"})
        torch.testing.assert_close(flip_only["flip"], result["flip"])

    def test_pair_statistics(self):
        stats = pair_statistics(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.zeros(4))
        self.assertEqual(stats["prediction_agreement"], .5)
        self.assertEqual(stats["both_wrong"], 1)
        self.assertEqual(stats["only_a_correct"], 1)
        self.assertEqual(stats["only_b_correct"], 1)


if __name__ == "__main__":
    unittest.main()
