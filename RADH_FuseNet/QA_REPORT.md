# Verification report

## Result

**PASS: 45 automated tests and the complete synthetic two-stage CLI demonstration.**

Verified on 2026-10-11 using Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.3.5 and SciPy 1.17.0, on Linux x86-64. CUDA was unavailable and was not tested. Tests use one CPU thread. No real participant data was used or uploaded.

This is a research reference implementation, not the authors' original training code or a reproduction of their reported benchmark scores. Passing these tests establishes software behavior on controlled inputs; it does not establish physiological validity, generalization, calibration, or driver-safety performance.

## Commands verified

From the package root, with dependencies installed:

```bash
python -m compileall -q radh tests scripts
python -m unittest discover -s tests -v
python -m radh demo --output runs/demo --device cpu --robustness
```

The demonstration was executed in an isolated temporary output directory using the unmodified packaged smoke configuration. The following interfaces were also exercised: all CLI help commands, checkpoint-based prediction, held-out evaluation, data loading, all three configuration files (`paper`, `smoke`, `wesad`), and a second small two-stage CLI run inside the test suite.

## Automated test coverage

- `test_data.py`: 14 tests. Subject-disjoint, deterministic and complete splits; rejection of explicit subject leakage; input/label/horizon validation; absent targets; nonfinite sample masking; training-only normalizer statistics; hidden-value invariance; normalizer serialization; constant/missing channels; gap-preserving resampling; bounded alignment without circular wrapping.
- `test_evidential.py`: 7 tests. Opinion and reliability-discount equations; probability/mass conservation; vacuous opinions; low-conflict conjunctive combination; high-conflict uncertainty transfer; stable reliability ordering; extreme finite evidence and gradients; rejection of extra environmental voting heads; NIG/Student-t variance consistency.
- `test_losses.py`: 7 tests. Student-t negative log likelihood checked against PyTorch's distribution; missing regression/classification targets and finite gradients; unequal shared/private widths; masked L1 reconstruction in decoder coordinates; absent contrastive negatives; bounded tiny-scale reconstruction and zero gradients for wholly absent source targets; reliability ordering and all-missing objectives.
- `test_model.py`: 14 tests. Default manuscript-scale and compact forward/backward passes; five sources and exactly four affective heads; nondivisible sequence lengths; constant/all-missing gradients; one missing source; all-missing/context-only uniform probabilities and total ignorance; neutral normalization for absent sources; exact invariance to masked values including NaN; nonfinite sample handling; hidden-patch reconstruction from observed context; checkpoint identity; configuration validation; custom external-sensor priors with humidity compensation disabled; causal selective-SSM and cross-scale prefix invariance; zero unavailable-node incidence.
- `test_integration.py`: 3 tests. CLI help; complete synthetic pretraining/fine-tuning with actual parameter changes; training-only checkpoint normalization; subject-held-out evaluation; exact reloaded prediction CSV identity; rejection of evaluation against a changed archive; all-missing/context-only invalid trajectory export; finite loss/optimizer updates with missing target combinations.

The default architecture test uses batch size 1 and 64 samples, with hidden width 96, shared/private widths 64/32, four encoder SSM blocks, state size 16, patch size 32 and temporal depths 1/2/6. Its 2,863,122 parameters perform a finite forward/backward pass. The compact test additionally verifies that every trainable parameter is connected to a differentiable composite objective and has a finite gradient.

The SSM/cross-scale causality tests apply to those modules. Whole-window quality estimates and spectral analysis still make the complete model an offline-window model.

## Full synthetic CLI run

The standard smoke configuration creates 30 simulated windows, each with five channels and 128 samples, from six artificial subject IDs. The forecast horizon is four. It uses four training subjects, one validation subject and one test subject; this is a software fixture, not a study cohort.

| Check | Pretraining | Supervised fine-tuning |
|---|---:|---:|
| Completed optimizer steps | 10 | 10 |
| Mean training objective | 0.797749 | 3.373319 |
| Mean reconstruction MAE | 0.789617 | 0.773499 |
| Maximum unclipped gradient norm | 0.170512 | 4.056082 |
| Sum of absolute parameter changes | 19.271412 | 36.997271 |

Parameter changes were measured between seeded initialization and the selected pretraining checkpoint, then between that checkpoint and the selected supervised checkpoint. They demonstrate that both stages update weights, rather than only running forward passes. Stage objectives differ and their numeric values should not be compared as a learning curve.

The run produced readable `best_pretrain.pt`, `best.pt`, `last.pt`, `split.json`, `config.json`, `history.json`, `metrics.json`, `predictions.csv`, and the explicitly synthetic data files. Checkpoints load with `weights_only=True`. Evaluation completed clean inputs plus Gaussian noise, spikes, contiguous dropout, drift, temporal shift, humidity variation, and each of five single-channel removal conditions.

Classification-only, regression-only and entirely unlabelled batches are covered at the loss/optimizer level. The standard training CLI deliberately requires some training and validation class labels because its supervised checkpoint-selection policy uses validation macro-F1. No alternate regression-only checkpoint-selection protocol is claimed.

## Defects caught and retested

- Constant-input normalization previously allowed a singular square-root gradient; the numerical floor now precedes the square root, and constant-input backward tests pass.
- Fully dropped source targets previously used an undefined near-zero normalization scale, inflating supervised reconstruction loss. Absent sources now have neutral normalization, their reconstruction targets are excluded, and transformed targets follow the model's numerical bounds. The full demonstration was rerun after the fix.
- Fully hidden patches previously risked decoding only a constant bias. Reconstruction now retains context propagated from observed samples while their graph incidence stays zero, with a regression test.
- Continuous predictions without physiological observations could be mistaken for valid forecasts. Such CSV rows now expose `trajectory_valid=False` with empty continuous mean/variance fields; classification remains uniform with uncertainty one.

## Not verified / scientific limitations

- No Dataset A data, original pretrained weights, author run logs or complete acquisition/annotation protocol were supplied. The manuscript's accuracy/F1 values have **not** been reproduced.
- WESAD configuration and no-humidity numerical behavior are tested, but no WESAD dataset was downloaded, trained or evaluated. The supplied adaptation is not evidence of WESAD performance.
- CUDA, mixed precision, distributed training, Windows/Anaconda installation, and the user's own computer were not tested.
- Large-cohort or long-sequence throughput, learned corruption monotonicity, predictive-interval coverage and real-world uncertainty calibration remain unvalidated.
- Unspecified acquisition filters, sensor thresholds, conflict transition choices and loss settings remain documented reference decisions; see `docs/METHOD_AND_ASSUMPTIONS.md`.

No smoke-test metric should be presented as a paper benchmark, medical conclusion, or safety certification.
