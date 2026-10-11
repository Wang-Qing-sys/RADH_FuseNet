import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from radh.data import load_npz, partition_indices, split_subjects, validate_split
from radh.preprocessing import RobustNormalizer, resample_masked, align_pair_by_correlation


class DataTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(23)
        self.x = self.rng.normal(size=(24, 5, 65)).astype(np.float32)
        self.mask = np.ones_like(self.x, dtype=bool)
        self.subjects = np.repeat([f"driver_{i:02d}" for i in range(8)], 3)

    def test_subject_split_disjoint_complete_and_reproducible(self):
        split = split_subjects(self.subjects, seed=41)
        again = split_subjects(self.subjects, seed=41)
        self.assertEqual(split, again)
        groups = [set(split[name]) for name in ("train", "val", "test")]
        self.assertTrue(all(groups))
        for i in range(len(groups)):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        self.assertEqual(set.union(*groups), set(self.subjects))
        indices = {name: partition_indices(self.subjects, split, name) for name in ("train", "val", "test")}
        covered = []
        for name in ("train", "val", "test"):
            selected = np.asarray(indices[name])
            covered.extend(selected.tolist())
            self.assertTrue(set(self.subjects[selected]).issubset(set(split[name])))
        self.assertEqual(sorted(covered), list(range(len(self.x))))

    def test_explicit_subject_leakage_is_rejected(self):
        split = split_subjects(self.subjects, seed=41)
        split["val"].append(split["train"][0])
        with self.assertRaisesRegex(ValueError, "leakage|overlap"):
            validate_split(split, self.subjects)

    def test_invalid_shapes_and_target_horizon_are_rejected(self):
        for changes in ({"x": self.x[:, :4]}, {"mask": self.mask[:, :, :-1]},
                        {"target": np.zeros((len(self.x), 2), dtype=np.float32)}):
            with self.subTest(changes=list(changes)):
                payload = dict(x=self.x, mask=self.mask, subject_ids=self.subjects)
                payload.update(changes)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "invalid.npz"
                    np.savez(path, **payload)
                    with self.assertRaises(ValueError):
                        load_npz(path, horizon=3, num_classes=5)

    def test_train_only_normalization_ignores_held_out_values(self):
        split = split_subjects(self.subjects, seed=19)
        train = np.asarray(partition_indices(self.subjects, split, "train"))
        first = RobustNormalizer.fit(self.x[train], self.mask[train])
        changed = self.x.copy()
        held_out = np.ones(len(changed), dtype=bool)
        held_out[train] = False
        changed[held_out] = 1e9
        second = RobustNormalizer.fit(changed[train], self.mask[train])
        self.assertEqual(json.dumps(first.to_dict(), sort_keys=True),
                         json.dumps(second.to_dict(), sort_keys=True))
        np.testing.assert_allclose(first.transform(self.x[train], self.mask[train]),
                                   second.transform(self.x[train], self.mask[train]))

    def test_masked_values_do_not_affect_fit_or_transform(self):
        mask = self.mask.copy()
        mask[:, 1, 4:30] = False
        changed = self.x.copy()
        changed[~mask] = 1e12
        original = RobustNormalizer.fit(self.x, mask)
        altered = RobustNormalizer.fit(changed, mask)
        self.assertEqual(json.dumps(original.to_dict(), sort_keys=True),
                         json.dumps(altered.to_dict(), sort_keys=True))
        normal = original.transform(self.x, mask)
        replacement = altered.transform(changed, mask)
        np.testing.assert_allclose(normal, replacement)
        self.assertTrue(np.isfinite(replacement).all())
        self.assertTrue((replacement[~mask] == 0).all())

    def test_normalizer_serialization_roundtrip(self):
        normalizer = RobustNormalizer.fit(self.x, self.mask)
        payload = json.loads(json.dumps(normalizer.to_dict()))
        loaded = RobustNormalizer.from_dict(payload)
        np.testing.assert_allclose(normalizer.transform(self.x, self.mask),
                                   loaded.transform(self.x, self.mask), rtol=0, atol=0)

    def test_constant_or_entirely_missing_channel_is_finite(self):
        x, mask = self.x.copy(), self.mask.copy()
        x[:, 0] = 7
        x[:, 4] = np.nan
        mask[:, 4] = False
        normalizer = RobustNormalizer.fit(x, mask)
        transformed = normalizer.transform(x, mask)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue((transformed[:, 4] == 0).all())

    def test_npz_supports_missing_regression_and_classification_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_labels.npz"
            np.savez(path, x=self.x, mask=self.mask, subject_ids=self.subjects,
                     y=np.full(len(self.x), -1, dtype=np.int64))
            data = load_npz(path, horizon=3, num_classes=5)
        self.assertEqual(data["x"].shape, self.x.shape)
        self.assertEqual(data["target"].shape, (len(self.x), 3))
        self.assertTrue(np.isnan(data["target"]).all())
        self.assertTrue((data["y"] == -1).all())

    def test_nonfinite_samples_become_unavailable(self):
        x = self.x.copy()
        x[0, 2, 7] = np.nan
        x[1, 1, 9] = np.inf
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.npz"
            np.savez(path, x=x, subject_ids=self.subjects,
                     y=np.arange(len(x), dtype=np.int64) % 5)
            data = load_npz(path, horizon=2, num_classes=5)
        self.assertFalse(bool(data["mask"][0, 2, 7]))
        self.assertFalse(bool(data["mask"][1, 1, 9]))
        self.assertTrue(np.isfinite(data["x"]).all())

    def test_npz_requires_subject_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe_split.npz"
            np.savez(path, x=self.x, y=np.arange(len(self.x)) % 5)
            with self.assertRaises((ValueError, KeyError)):
                load_npz(path, horizon=3, num_classes=5)

    def test_invalid_class_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_labels.npz"
            np.savez(path, x=self.x, subject_ids=self.subjects,
                     y=np.full(len(self.x), 5, dtype=np.int64))
            with self.assertRaises(ValueError):
                load_npz(path, horizon=3, num_classes=5)


class PreprocessingTests(unittest.TestCase):
    def test_resampling_preserves_long_gap(self):
        source_hz, target_hz = 100.0, 50.0
        x = np.sin(np.arange(200) * 0.12).astype(np.float32)[None]
        mask = np.ones_like(x, dtype=bool)
        mask[:, 40:120] = False
        x[:, 40:120] = 1e12
        output, observed = resample_masked(x, mask, source_hz, target_hz)
        self.assertEqual(output.shape, (1, 100))
        self.assertFalse(observed[:, 20:60].any())
        self.assertTrue((output[:, 20:60] == 0).all())
        self.assertTrue(np.isfinite(output).all())

    def test_resampling_does_not_read_hidden_values(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(5, 128)).astype(np.float32)
        mask = np.ones_like(x, dtype=bool)
        mask[:, 30:80] = False
        altered = x.copy()
        altered[~mask] = np.nan
        y1, m1 = resample_masked(x, mask, 128, 64)
        y2, m2 = resample_masked(altered, mask, 128, 64)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_allclose(y1, y2)

    def test_alignment_is_bounded_without_circular_wrap(self):
        rng = np.random.default_rng(33)
        reference = rng.normal(size=100).astype(np.float32)
        source = np.zeros_like(reference)
        source[5:] = reference[:-5]
        mask = np.ones_like(reference, dtype=bool)
        aligned, observed, lag = align_pair_by_correlation(reference, source, mask, mask, 8)
        self.assertEqual(lag, -5)
        np.testing.assert_allclose(aligned[:-5], reference[:-5])
        self.assertFalse(observed[-5:].any())
        self.assertTrue((aligned[-5:] == 0).all())


if __name__ == "__main__":
    unittest.main()
