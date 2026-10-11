import unittest

import torch

from radh.evidential import nig_predictive
from radh.losses import (RADHLoss, evidential_classification_loss, nig_regression_loss,
                        normal_inverse_gamma_nll, physiological_infonce,
                        shared_private_orthogonality)


torch.set_num_threads(1)


class LossTests(unittest.TestCase):
    def test_nig_nll_matches_distribution(self):
        nig = {"mu": torch.randn(2, 3), "nu": torch.rand(2, 3) + .1,
               "alpha": torch.rand(2, 3) + 1.1, "beta": torch.rand(2, 3) + .1}
        target = torch.randn(2, 3)
        pred = nig_predictive(nig)
        expected = -torch.distributions.StudentT(pred["df"], pred["mean"], pred["scale"]).log_prob(target)
        torch.testing.assert_close(normal_inverse_gamma_nll(nig, target), expected)

    def test_nan_regression_targets_and_missing_labels_have_finite_gradients(self):
        nig = {key: torch.full((2, 3), value, requires_grad=True)
               for key, value in {"mu": 0., "nu": 1., "alpha": 2., "beta": 1.}.items()}
        loss = nig_regression_loss(nig, torch.full((2, 3), float("nan")))
        self.assertEqual(float(loss), 0)
        loss.backward()
        for value in nig.values():
            self.assertTrue(torch.isfinite(value.grad).all())
        alpha = torch.full((2, 5), 2., requires_grad=True)
        loss = evidential_classification_loss(alpha, torch.full((2,), -1))
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(alpha.grad).all())

    def test_orthogonality_accepts_different_widths_and_missing_tokens(self):
        shared = torch.randn(2, 5, 4, 8, requires_grad=True)
        private = torch.randn(2, 5, 4, 3, requires_grad=True)
        mask = torch.zeros(2, 5, 4, dtype=torch.bool)
        self.assertEqual(float(shared_private_orthogonality(shared, private, mask)), 0)
        mask[:, :3] = True
        loss = shared_private_orthogonality(shared, private, mask)
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertTrue(torch.isfinite(shared.grad).all())
        self.assertTrue(torch.isfinite(private.grad).all())

    def test_reconstruction_is_masked_l1_in_decoder_coordinates(self):
        reconstruction = torch.zeros(1, 5, 2, requires_grad=True)
        raw_target = torch.full((1, 5, 2), 100.)
        corrected = torch.full((1, 5, 2), 10.)
        outputs = {"alpha": torch.ones(1, 5), "reconstruction": reconstruction,
                   "reconstruction_center": torch.full((1, 5, 1), 4.),
                   "reconstruction_scale": torch.full((1, 5, 1), 2.),
                   "availability": torch.ones(1, 5, dtype=torch.bool)}
        mask = torch.ones_like(reconstruction, dtype=torch.bool)
        selected = torch.zeros_like(mask)
        selected[:, :, 0] = True
        batch = {"x": raw_target, "mask": mask, "target_mask": mask,
                 "reconstruction_target": raw_target, "reconstruction_mask": selected}
        losses = RADHLoss()(outputs, batch, stage="pretrain", clean_outputs={"corrected_input": corrected})
        self.assertEqual(float(losses["reconstruction"]), 3.)
        losses["total"].backward()
        self.assertTrue((reconstruction.grad[:, :, 1] == 0).all())
        self.assertTrue((reconstruction.grad[:, :, 0] != 0).all())

    def test_missing_contrastive_negatives_are_explicit(self):
        shared = torch.randn(1, 5, 4, 8, requires_grad=True)
        loss, pairs = physiological_infonce(shared, torch.ones(1, 5, 4, dtype=torch.bool))
        self.assertEqual(pairs, 0)
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(shared.grad).all())

    def test_wholly_missing_reconstruction_ignored_and_tiny_scale_bounded(self):
        reconstruction = torch.zeros(1, 5, 2, requires_grad=True)
        target = torch.ones_like(reconstruction)
        availability = torch.tensor([[False, True, False, False, False]])
        outputs = {"alpha": torch.ones(1, 5), "reconstruction": reconstruction,
                   "reconstruction_center": torch.zeros(1, 5, 1),
                   "reconstruction_scale": torch.full((1, 5, 1), 1e-4),
                   "availability": availability}
        batch = {"x": target, "mask": torch.ones_like(target, dtype=torch.bool),
                 "reconstruction_target": target}
        losses = RADHLoss()(outputs, batch)
        self.assertEqual(float(losses["reconstruction"]), 50.)
        losses["total"].backward()
        self.assertTrue((reconstruction.grad[:, 0] == 0).all())
        self.assertTrue((reconstruction.grad[:, 1] != 0).all())

    def test_hinge_ordering_and_all_missing_objective(self):
        alpha = torch.ones(2, 5, requires_grad=True)
        full = torch.ones(2, 5, dtype=torch.bool)
        outputs = {"alpha": alpha, "availability": full,
                   "reliability": torch.full((2, 5), .9, requires_grad=True)}
        mild = {"reliability": torch.full((2, 5), .5)}
        strong = {"reliability": torch.full((2, 5), .1)}
        batch = {"x": torch.zeros(2, 5, 4), "mask": torch.ones(2, 5, 4, dtype=torch.bool),
                 "y": torch.tensor([0, 1])}
        objective = RADHLoss()
        loss = objective(outputs, batch, mild_outputs=mild, strong_outputs=strong)
        self.assertEqual(float(loss["ranking"]), 0)
        bad = {"reliability": torch.ones(2, 5)}
        loss = objective(outputs, batch, strong_outputs=bad)
        self.assertGreater(float(loss["ranking"]), 0)
        outputs["availability"] = torch.zeros_like(full)
        loss = objective(outputs, batch)
        self.assertEqual(float(loss["total"]), 0)
        loss["total"].backward()
        self.assertTrue(torch.isfinite(alpha.grad).all())


if __name__ == "__main__":
    unittest.main()
