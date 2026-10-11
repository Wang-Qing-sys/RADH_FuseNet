"""Architecture, missingness, causal-block, and checkpoint regression tests."""
import io
import unittest

import torch

from radh.blocks import CrossScaleTemporalFusion, DynamicHypergraphLayer, SelectiveSSMBlock
from radh.model import ModelConfig, RADHFuseNet

# Small scans are slower with many OpenMP threads than with one CPU thread.
torch.set_num_threads(1)


def compact_config(**changes):
    values = dict(hidden_dim=16, shared_dim=8, private_dim=8, state_dim=4,
                  ssm_depth=1, temporal_depths=(1, 1, 1), patch_size=8,
                  n_fft=16, dropout=0.0)
    values.update(changes)
    return ModelConfig(**values)


def assert_finite_tree(testcase, value, prefix="output"):
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        testcase.assertTrue(torch.isfinite(value).all().item(), f"Nonfinite tensor: {prefix}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(testcase, item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_tree(testcase, item, f"{prefix}[{index}]")


def differentiable_objective(out):
    # Exercise every encoder branch, relational/temporal stack, and output head.
    nig = out["nig"]
    return (-out["probabilities"][:, 0].clamp_min(1e-8).log().mean()
            + out["reconstruction"].square().mean()
            + out["reliability"].mean()
            + out["shared"].square().mean() + out["private"].square().mean()
            + sum(value.square().mean() for value in nig.values()))


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)

    def test_compact_odd_sequence_forward_backward_and_four_opinions(self):
        config = compact_config()
        model = RADHFuseNet(config)
        x = torch.randn(2, 5, 65, requires_grad=True)
        out = model(x)
        assert_finite_tree(self, out)
        self.assertEqual(out["probabilities"].shape, (2, 5))
        self.assertEqual(out["evidence_heads"].shape, (2, 4, 5))
        self.assertEqual(out["reliability"].shape, (2, 5))
        self.assertEqual(out["reconstruction"].shape, x.shape)
        self.assertEqual(out["shared"].shape[:3], (2, 5, 9))
        self.assertEqual(out["head_reliability"].shape, (2, 4))
        torch.testing.assert_close(out["probabilities"].sum(-1), torch.ones(2))
        self.assertTrue((out["nig"]["nu"] > 0).all())
        self.assertTrue((out["nig"]["alpha"] > 1).all())
        self.assertTrue((out["nig"]["beta"] > 0).all())
        differentiable_objective(out).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        gradients = {name: p.grad for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(gradients)
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, f"Disconnected parameter: {name}")
            self.assertTrue(torch.isfinite(gradient).all(), f"Nonfinite gradient: {name}")
        self.assertTrue(any(gradient.abs().sum().item() > 0 for gradient in gradients.values()))

    def test_default_manuscript_architecture_forward_backward(self):
        config = ModelConfig()
        self.assertEqual(config.hidden_dim, 96)
        self.assertEqual(config.shared_dim, 64)
        self.assertEqual(config.private_dim, 32)
        self.assertEqual(config.state_dim, 16)
        self.assertEqual(config.ssm_depth, 4)
        self.assertEqual(tuple(config.temporal_depths), (1, 2, 6))
        self.assertEqual(config.patch_size, 32)
        self.assertEqual(config.n_fft, 64)
        model = RADHFuseNet(config)
        out = model(torch.randn(1, 5, 64))
        assert_finite_tree(self, out)
        self.assertEqual(out["evidence_heads"].shape, (1, 4, 5))
        differentiable_objective(out).backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum().item() > 0 for g in gradients))

    def test_all_missing_and_context_only_are_exact_ignorance(self):
        model = RADHFuseNet(compact_config()).eval()
        x = torch.randn(2, 5, 65)
        for context_only in (False, True):
            with self.subTest(context_only=context_only):
                mask = torch.zeros_like(x, dtype=torch.bool)
                if context_only:
                    mask[:, 3:] = True
                out = model(x, mask=mask)
                assert_finite_tree(self, out)
                torch.testing.assert_close(out["probabilities"], torch.full((2, 5), 0.2), atol=1e-7, rtol=0)
                torch.testing.assert_close(out["uncertainty"], torch.ones(2), atol=1e-7, rtol=0)
                torch.testing.assert_close(out["head_reliability"], torch.zeros(2, 4), atol=0, rtol=0)
                self.assertTrue((out["reliability"][:, :3] == 0).all())
                torch.testing.assert_close(out["reconstruction_center"][:, :3],
                                           torch.zeros_like(out["reconstruction_center"][:, :3]))
                torch.testing.assert_close(out["reconstruction_scale"][:, :3],
                                           torch.ones_like(out["reconstruction_scale"][:, :3]))

    def test_fully_hidden_patch_reconstructs_from_observed_context(self):
        model = RADHFuseNet(compact_config()).eval()
        x = torch.randn(1, 5, 32)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 0, 8:16] = False
        changed = x.clone()
        changed[:, 0, :8] = torch.randn_like(changed[:, 0, :8]) * 3
        first, second = model(x, mask), model(changed, mask)
        self.assertEqual(float(first["patch_availability"][0, 0, 1]), 0)
        self.assertFalse(torch.equal(first["reconstruction"][0, 0, 8:16],
                                     second["reconstruction"][0, 0, 8:16]))

    def test_external_sensor_configuration_without_humidity(self):
        config = compact_config(num_classes=3, humidity_compensation=False,
                                prior_groups=((0, 1), (0, 1, 2)),
                                quality_bands=((0.7, 4), (0, 1), (.05, .8), (0, 1), (0, 1)))
        model = RADHFuseNet(config)
        x = torch.randn(1, 5, 33)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 3:] = False
        out = model(x, mask)
        assert_finite_tree(self, out)
        self.assertEqual(out["probabilities"].shape, (1, 3))
        self.assertEqual(out["evidence_heads"].shape, (1, 4, 3))
        self.assertTrue((out["compensation"] == 0).all())
        differentiable_objective(out).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_missing_source_has_zero_reliability(self):
        model = RADHFuseNet(compact_config()).eval()
        x = torch.randn(2, 5, 64)
        mask = torch.ones_like(x, dtype=torch.bool)
        mask[:, 1] = False
        out = model(x, mask=mask)
        assert_finite_tree(self, out)
        self.assertTrue((out["reliability"][:, 1] == 0).all())
        self.assertTrue((out["patch_availability"][:, 1] == 0).all())
        self.assertTrue((out["head_reliability"][:, 1] == 0).all())

    def test_masked_values_and_nonfinite_hidden_samples_cannot_change_output(self):
        model = RADHFuseNet(compact_config()).eval()
        x = torch.randn(2, 5, 65)
        mask = torch.rand_like(x) > 0.25
        mask[:, 1] = False
        changed = x.clone()
        changed[~mask] = float("nan")
        first = model(x, mask=mask)
        second = model(changed, mask=mask)
        assert_finite_tree(self, second)
        for key in ("probabilities", "uncertainty", "reliability", "compensation", "reconstruction"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0, msg=key)
        for key in ("mu", "nu", "alpha", "beta"):
            torch.testing.assert_close(first["nig"][key], second["nig"][key], rtol=0, atol=0)

    def test_unmasked_nonfinite_samples_are_treated_as_missing(self):
        model = RADHFuseNet(compact_config()).eval()
        x = torch.randn(1, 5, 32)
        x[0, 2, 7] = float("nan")
        x[0, 4, 5] = float("inf")
        out = model(x)
        explicit = model(torch.nan_to_num(x), mask=torch.isfinite(x))
        assert_finite_tree(self, out)
        torch.testing.assert_close(out["probabilities"], explicit["probabilities"], rtol=0, atol=0)

    def test_constant_observed_and_all_missing_backward_remain_finite(self):
        for value, available in ((0.0, True), (1.0, True), (100.0, True), (0.0, False)):
            with self.subTest(value=value, available=available):
                model = RADHFuseNet(compact_config())
                x = torch.full((1, 5, 17), value, requires_grad=True)
                mask = torch.full_like(x, available, dtype=torch.bool)
                out = model(x, mask)
                assert_finite_tree(self, out)
                differentiable_objective(out).backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                    for p in model.parameters()))

    def test_state_dict_save_reload_preserves_predictions(self):
        config = compact_config()
        model = RADHFuseNet(config).eval()
        x = torch.randn(2, 5, 65)
        first = model(x)
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        loaded = RADHFuseNet(config).eval()
        loaded.load_state_dict(torch.load(checkpoint, weights_only=True))
        second = loaded(x)
        for key in ("probabilities", "uncertainty", "reliability", "conflict"):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        for key in first["nig"]:
            torch.testing.assert_close(first["nig"][key], second["nig"][key], rtol=0, atol=0)

    def test_configuration_and_input_validation(self):
        for changes in ({"hidden_dim": 0}, {"state_dim": 0}, {"patch_size": 0},
                        {"n_fft": 1}, {"num_classes": 1}, {"dropout": 1.2},
                        {"temporal_depths": (1, 2)}):
            with self.subTest(changes=changes):
                with self.assertRaises((ValueError, TypeError)):
                    RADHFuseNet(compact_config(**changes))
        model = RADHFuseNet(compact_config())
        with self.assertRaises(ValueError):
            model(torch.randn(2, 4, 64))
        with self.assertRaises(ValueError):
            model(torch.randn(2, 5, 64), mask=torch.ones(2, 5, 63, dtype=torch.bool))


class BlockTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_selective_ssm_is_future_invariant(self):
        model = SelectiveSSMBlock(8, state_dim=4, expansion=2, dropout=0).eval()
        x = torch.randn(2, 19, 8)
        changed = x.clone()
        changed[:, 10:] = torch.randn_like(changed[:, 10:]) * 100
        first, second = model(x), model(changed)
        torch.testing.assert_close(first[:, :10], second[:, :10], rtol=0, atol=0)
        first.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cross_scale_temporal_module_is_future_invariant(self):
        model = CrossScaleTemporalFusion(8, (1, 2, 2), 4, 2, 0).eval()
        for length in (1, 3, 7, 17):
            with self.subTest(length=length):
                x = torch.randn(2, length, 8)
                reliability = torch.full((2,), 0.7)
                first, gates = model(x, reliability)
                self.assertEqual(first.shape, x.shape)
                self.assertEqual(gates.shape, (2, length, 3))
                torch.testing.assert_close(gates.sum(-1), torch.ones(2, length))
                if length > 1:
                    cutoff = length // 2
                    changed = x.clone()
                    changed[:, cutoff:] += 100
                    second, _ = model(changed, reliability)
                    torch.testing.assert_close(first[:, :cutoff], second[:, :cutoff], rtol=0, atol=0)

    def test_unavailable_hypergraph_nodes_have_zero_incidence(self):
        layer = DynamicHypergraphLayer(8, top_k=2, dropout=0)
        x = torch.randn(2, 5, 3, 8)
        reliability = torch.ones(2, 5)
        available = torch.ones(2, 5, 3)
        reliability[:, 1] = 0
        available[:, 3, 1] = 0
        node_ids, _, incidence, _ = layer._incidences(x, reliability, available)
        valid_nodes = (reliability[:, :, None] * available).reshape(2, -1)
        inactive = valid_nodes.gather(1, node_ids) == 0
        self.assertTrue(inactive.any())
        self.assertTrue((incidence[inactive] == 0).all())
        output = layer(x, reliability, available)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue((output[:, 3, 1] == 0).all())
        all_absent = layer(x, torch.zeros_like(reliability), torch.zeros_like(available))
        self.assertTrue(torch.isfinite(all_absent).all())
        self.assertTrue((all_absent == 0).all())


if __name__ == "__main__":
    unittest.main()
