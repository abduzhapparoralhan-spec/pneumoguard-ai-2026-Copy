# Preserved measured results

**Full run: CNN accuracy 82.85% on 624 test images; abstention coverage 100%, with no improvement.**

These are the 17 files from [PneumoGuard_full_20261115T094604_549333Z.zip](../PneumoGuard_full_20261115T094604_549333Z.zip), copied byte for byte. The ZIP, CNN implementation, notebook and trained weights were preserved. No training, new inference, threshold tuning or figure regeneration was performed for this update.

## Evidence to read

- [Original measured report](report.md), [metrics JSON](metrics.json), [comparison CSV](comparison.csv), and [all 624 test predictions](test_predictions.csv).
- [Frozen policy](policy.json), [validation predictions](validation_predictions.csv), and [121-candidate validation policy grid](validation_policy_grid.csv).
- [Configuration](config.json), [training manifest](training_manifest.json), [20-epoch training history](training_history.csv), [split manifest](split_manifest.json), and [recorded runtime](environment.json).
- [Trained checkpoint](best_model.pt): preserved from the archive; its hash matches the training manifest. Load only with the matching existing architecture and `weights_only=True`, as shown in the [README](../README.md).
- [Confusion matrices](figures/confusion_matrices.png), [training and validation trade-off](figures/training_and_validation_tradeoff.png), [five actual mistakes](figures/failure_examples.png), and [failure example table](failure_examples.csv).

The CNN made 517 correct predictions, 5 false negatives and 102 false positives. The validation-selected thresholds are both 0.5, so all 624 images receive the same decisions as forced choice and none require review. The automatic error rate is 107/624 = 17.15%. Validation error was 18/524 = 3.44%; the 5% target is an empirical validation setting, not a test or clinical guarantee.

The original report contains a general warning that accepted-only metrics do not describe the whole test set. In this particular run, the accepted subset equals the whole test set because coverage is 100%. Both metric rows therefore coincide. The original report is retained unchanged.

## Provenance and limits of this audit

[provenance.json](provenance.json) records the source repository commit, archive Git blob SHA, archive SHA-256 and SHA-256 of every archived file. The archive SHA-256 is:

```text
4db70b827beba7f0691c183985751d7a4b0339e33f327f56d8ee9b20a8594655
```

The checkpoint SHA-256 is:

```text
66b825a4b1a216bdf67e237e1f08b84453dcbfa7bfd69f537d6f3562238243d1
```

The archive records full mode (`smoke: false`), seed 2026, CPU, Python 3.13.18, and best epoch 20. Its configuration retains the original Colab output path; that is provenance, not a required local path. Metrics and validation-policy selection were independently checked against the saved CSVs. Source data were not downloaded again, and checkpoint inference was not rerun. The original archive contains neither an executed notebook nor a training-time source-code commit, so this audit is not an independent end-to-end reproduction. See the [verification record](../docs/verification.md).

The archived split audit records 8 distinct shared image hashes between training and validation, with 0 shared hashes between training–test or validation–test. Validation overlap and reuse for two selections can make validation estimates optimistic. Patient IDs are unavailable; patient independence cannot be verified. The source arrays were not independently audited again in this update.

## Attribution and medical limits

Images in the failure figure are enlarged 28×28 PneumoniaMNIST derivatives of Kermany et al.'s pediatric chest X-ray data, licensed **CC BY 4.0**. Enlarging adds no information. Preserve the [dataset, source-paper, library and AI-tool credits](../docs/sources.md) when sharing these artifacts. The five displayed mistakes are the first by test index and are all false positives; they do not represent every failure type. A high model score is not a calibrated probability or a diagnosis, and disagreement with a reference annotation does not establish a clinical cause.

One seed, a small pediatric benchmark, restricted source population, low resolution, class imbalance, and possible dataset artifacts limit generalization to adults, other hospitals or devices. No clinical validation, prospective benefit, or human-review accuracy is claimed. No private scans are accepted.

**This project is a prototype and is not medical advice. Predictions are for research and educational purposes only and must not be used for clinical decisions.**
