import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from radh.data import load_npz, partition_indices
from radh.losses import RADHLoss
from radh.model import ModelConfig, RADHFuseNet
from radh.preprocessing import RobustNormalizer
from radh.train import _forward_loss

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class IntegrationTests(unittest.TestCase):
    def run_cli(self, *arguments, success=True):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["OMP_NUM_THREADS"] = "1"
        result = subprocess.run([sys.executable, "-m", "radh", *map(str, arguments)],
                                cwd=ROOT, env=environment, text=True, capture_output=True,
                                timeout=180)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, "Invalid operation unexpectedly succeeded")
        return result

    def test_command_line_help(self):
        for command in ((), ("demo",), ("train",), ("evaluate",), ("predict",), ("make-demo-data",)):
            with self.subTest(command=command):
                result = self.run_cli(*command, "--help")
                self.assertIn("usage:", result.stdout)

    def test_two_stage_cli_demo_updates_weights_and_roundtrips_predictions(self):
        config = json.loads((ROOT / "configs" / "smoke.json").read_text())
        config["model"].update(dropout=0.0, horizon=2)
        config["training"].update(cpu_threads=1, batch_size=2, seed=37,
                                  pretrain_epochs=1, finetune_epochs=1)
        config["demo"] = {"samples": 32, "subjects": 3, "windows_per_subject": 2}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            configuration = Path(directory) / "config.json"
            configuration.write_text(json.dumps(config))
            self.run_cli("demo", "--output", output, "--config", configuration, "--device", "cpu")
            for name in ("best_pretrain.pt", "best.pt", "last.pt", "history.json", "metrics.json",
                         "synthetic.npz", "predictions.csv", "split.json"):
                self.assertTrue((output / name).is_file(), name)
            history = json.loads((output / "history.json").read_text())
            self.assertEqual([row["stage"] for row in history], ["pretrain", "supervised"])
            for row in history:
                self.assertGreater(row["optimizer_steps"], 0)
                self.assertGreater(row["max_unclipped_gradient_norm"], 0)
                self.assertTrue(all(math.isfinite(value) for value in row["train"].values()))
            pretrain = torch.load(output / "best_pretrain.pt", weights_only=True)
            final = torch.load(output / "best.pt", weights_only=True)
            torch.manual_seed(config["training"]["seed"])
            initial = RADHFuseNet(ModelConfig(**config["model"])).state_dict()
            def difference(left, right):
                return sum(float((left[key] - right[key]).abs().sum())
                           for key in left if left[key].is_floating_point())
            self.assertGreater(difference(initial, pretrain["model_state"]), 0)
            self.assertGreater(difference(pretrain["model_state"], final["model_state"]), 0)
            self.assertEqual(final["normalizer_fit_partition"], "train")
            self.assertTrue(final["synthetic_demo"])
            data = load_npz(output / "synthetic.npz", horizon=2, num_classes=5)
            train = partition_indices(data["subject_ids"], final["split"], "train")
            expected = RobustNormalizer.fit(data["x"][train], data["mask"][train])
            self.assertEqual(final["normalizer"], expected.to_dict())
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertEqual(metrics["partition"], "test")
            self.assertTrue(metrics["synthetic_demo"])
            self.assertEqual(set(metrics["subject_ids"]), set(final["split"]["test"]))
            prediction = output / "prediction_reload.csv"
            self.run_cli("predict", "--data", output / "synthetic.npz", "--checkpoint", output / "best.pt",
                         "--output", prediction, "--partition", "test", "--batch-size", "2", "--threads", "1")
            self.assertEqual(prediction.read_text(), (output / "predictions.csv").read_text())
            with prediction.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertAlmostEqual(sum(float(row[f"probability_{i}"]) for i in range(5)), 1, places=6)
                self.assertGreater(float(row["predictive_variance_0"]), 0)
            missing = {key: value.copy() for key, value in data.items()}
            missing["mask"][0] = False
            missing["mask"][1, :3] = False
            missing["x"][~missing["mask"]] = 0
            missing_path = output / "missing_physiology.npz"
            np.savez(missing_path, **missing)
            missing_csv = output / "missing_predictions.csv"
            self.run_cli("predict", "--data", missing_path, "--checkpoint", output / "best.pt",
                         "--output", missing_csv, "--threads", "1", "--batch-size", "2")
            with missing_csv.open() as handle:
                missing_rows = list(csv.DictReader(handle))
            for row in missing_rows[:2]:
                self.assertIn(row["trajectory_valid"].lower(), ("false", "0"))
                self.assertEqual(row["mean_0"], "")
                self.assertEqual(row["predictive_variance_0"], "")
                self.assertAlmostEqual(float(row["uncertainty"]), 1, places=6)
                for index in range(5):
                    self.assertAlmostEqual(float(row[f"probability_{index}"]), .2, places=6)
            changed = dict(data)
            changed["x"] = data["x"].copy()
            changed["x"][0, 0, 0] += 1
            np.savez(output / "different_data.npz", **changed)
            rejected = self.run_cli("evaluate", "--data", output / "different_data.npz",
                                    "--checkpoint", output / "best.pt", "--output", output / "invalid_metrics.json",
                                    "--threads", "1", success=False)
            self.assertIn("differs from the training NPZ", rejected.stderr)

    def test_missing_target_modes_are_finite_with_optimizer_updates(self):
        values = dict(hidden_dim=16, shared_dim=8, private_dim=8, state_dim=4,
                      ssm_depth=1, temporal_depths=(1, 1, 1), patch_size=8,
                      n_fft=16, dropout=0.0, horizon=3, hypergraph_layers=1)
        for mode in ("classification_only", "regression_only", "unlabelled"):
            with self.subTest(mode=mode):
                torch.manual_seed(89)
                model = RADHFuseNet(ModelConfig(**values))
                criterion = RADHLoss()
                x = torch.randn(2, 5, 32)
                batch = {"x": x, "mask": torch.ones_like(x, dtype=torch.bool),
                         "y": torch.tensor([0, 2]), "target": torch.randn(2, 3)}
                if mode != "classification_only":
                    batch["y"][:] = -1
                if mode != "regression_only":
                    batch["target"][:] = float("nan")
                before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                losses = _forward_loss(model, criterion, batch,
                                       {"corruption_severity": .2, "modality_dropout": .1},
                                       "supervised", 1)
                self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                if mode != "regression_only":
                    self.assertEqual(float(losses["regression"].detach()), 0)
                if mode != "classification_only":
                    self.assertEqual(float(losses["classification"].detach()), 0)
                losses["total"].backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
                torch.optim.AdamW(model.parameters(), lr=2e-4).step()
                self.assertTrue(any(not torch.equal(before[name], p) for name, p in model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
