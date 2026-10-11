import unittest

import torch

from radh.evidential import (combine_opinions, discount_opinion, evidence_to_opinion,
                             fuse_evidence, nig_predictive, nig_predictive_variance)


torch.set_num_threads(1)


class EvidentialTests(unittest.TestCase):
    def test_opinion_equations_and_discount(self):
        evidence = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        belief, uncertainty = evidence_to_opinion(evidence)
        torch.testing.assert_close(belief, evidence / 9)
        torch.testing.assert_close(uncertainty, torch.tensor([1 / 3], dtype=torch.float64))
        discounted, u = discount_opinion(belief, uncertainty, torch.tensor([0.25]))
        torch.testing.assert_close(discounted, belief * 0.25)
        torch.testing.assert_close(u, 1 - 0.25 * (1 - uncertainty))
        torch.testing.assert_close(discounted.sum(-1) + u, torch.ones_like(u))

    def test_all_unavailable_is_vacuous_and_backward_is_finite(self):
        evidence = torch.rand(3, 4, 5, requires_grad=True)
        reliability = torch.zeros(3, 4, requires_grad=True)
        result = fuse_evidence(evidence, reliability)
        torch.testing.assert_close(result["alpha"], torch.ones(3, 5))
        torch.testing.assert_close(result["uncertainty"], torch.ones(3))
        torch.testing.assert_close(result["conflict"], torch.zeros(3))
        torch.testing.assert_close(result["probabilities"], torch.full((3, 5), 0.2))
        result["alpha"].sum().backward()
        self.assertTrue(torch.isfinite(evidence.grad).all())
        self.assertTrue(torch.isfinite(reliability.grad).all())

    def test_conflict_transferred_to_uncertainty(self):
        a, ua = evidence_to_opinion(torch.tensor([[1000.0, 0.0, 0.0]]))
        b, ub = evidence_to_opinion(torch.tensor([[0.0, 1000.0, 0.0]]))
        conflict = combine_opinions(a, ua, b, ub, threshold=0.4)
        agreement = combine_opinions(a, ua, a, ua, threshold=0.4)
        self.assertTrue(conflict["used_yager"].all())
        self.assertGreater(float(conflict["uncertainty"]), 0.9)
        self.assertGreater(float(conflict["uncertainty"]), float(agreement["uncertainty"]))
        torch.testing.assert_close(conflict["belief"].sum(-1) + conflict["uncertainty"], torch.ones(1))

    def test_low_conflict_normalized_conjunctive_equation(self):
        a, ua = evidence_to_opinion(torch.tensor([[2.0, 0.1, 0.2]], dtype=torch.float64))
        b, ub = evidence_to_opinion(torch.tensor([[3.0, 0.2, 0.1]], dtype=torch.float64))
        result = combine_opinions(a, ua, b, ub, threshold=0.9)
        k = (a.sum(-1) * b.sum(-1) - (a * b).sum(-1))
        expected = (a * b + a * ub[:, None] + b * ua[:, None]) / (1 - k[:, None])
        torch.testing.assert_close(result["belief"], expected)
        torch.testing.assert_close(result["uncertainty"], ua * ub / (1 - k))

    def test_sorted_and_finite_large_evidence(self):
        evidence = torch.full((2, 4, 5), 1e20, requires_grad=True)
        reliability = torch.tensor([[0.2, 0.8, 0.6, 0.4], [1., 1., 1., 1.]], requires_grad=True)
        result = fuse_evidence(evidence, reliability)
        self.assertEqual(result["fusion_order"][0].tolist(), [1, 2, 3, 0])
        self.assertEqual(result["fusion_order"][1].tolist(), [0, 1, 2, 3])
        for value in result.values():
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(result["belief"].sum(-1) + result["uncertainty"], torch.ones(2))
        result["alpha"].sum().backward()
        self.assertTrue(torch.isfinite(evidence.grad).all())
        self.assertTrue(torch.isfinite(reliability.grad).all())

    def test_reject_extra_environmental_voters(self):
        with self.assertRaises(ValueError):
            fuse_evidence(torch.ones(2, 5, 5), torch.ones(2, 5))

    def test_nig_matches_student_t(self):
        nig = {"mu": torch.zeros(2, 3), "nu": torch.full((2, 3), 2.),
               "alpha": torch.full((2, 3), 3.), "beta": torch.full((2, 3), 4.)}
        result = nig_predictive(nig)
        distribution = torch.distributions.StudentT(result["df"], result["mean"], result["scale"])
        torch.testing.assert_close(distribution.variance, result["variance"])
        torch.testing.assert_close(result["variance"], nig_predictive_variance(nig["nu"], nig["alpha"], nig["beta"]))
        torch.testing.assert_close(result["variance"], result["aleatoric_variance"] + result["epistemic_variance"])


if __name__ == "__main__":
    unittest.main()
