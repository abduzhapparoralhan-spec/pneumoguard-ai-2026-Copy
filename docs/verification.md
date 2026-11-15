# Verification record

Development and results-integration date: November 15, 2026. The original implementation was added in merged PR #1. This update continues the existing project at `main` commit `6b07589c715e5a9567e678a52924349bf9f7d251`, where the genuine Colab results ZIP was already present. It preserves the implementation, notebook, both existing ZIPs and trained checkpoint.

## Measured-run artifact checks

Source: `PneumoGuard_full_20261115T094604_549333Z.zip`, archive SHA-256 `4db70b827beba7f0691c183985751d7a4b0339e33f327f56d8ee9b20a8594655`.

- ZIP integrity passed. All **17 archived files** were extracted to `results/` and compared byte for byte; file hashes are in [provenance.json](../results/provenance.json).
- Checkpoint and policy SHA-256 values match `training_manifest.json`. The checkpoint was not deserialized or replaced during this update.
- The run records `smoke: false`, seed 2026, 55,041 parameters, 20 training epochs and best validation-loss checkpoint epoch 20. The recorded training runtime is in `results/environment.json`; it is distinct from the earlier development environment below.
- All **624 test rows** were independently checked: IDs, class counts (234 Normal, 390 Pneumonia), score thresholds, predictions, acceptance and correctness flags. All baseline, forced-CNN, selective-CNN and `comparison.csv` metrics agree with recomputation from the saved CSVs.
- CNN: **517/624 correct = 82.85% accuracy**, confusion matrix `[[132,102],[5,385]]`, balanced accuracy 77.56%, Pneumonia precision 79.06%, recall 98.72%, F1 87.80%, 5 false negatives and 102 false positives.
- All **524 validation rows** and **121 grid candidates** were checked. The documented maximum-coverage rule selects `low = high = 0.5`; validation error is **18/524 = 3.44%**.
- The frozen policy makes **624 automatic decisions, 0 reviews, 100% coverage**. All outputs equal forced choice. Test automatic error is **107/624 = 17.15%**. Abstention gives **no improvement** in this run; the validation target did not guarantee test performance.
- The five saved failure examples match the first five mistakes in test-index order. They are all false positives, not a representative sample of all mistakes.
- All three PNGs passed image decoding and were visually inspected. They are original archived figures; no plots were regenerated.
- Final static checks passed: Python module syntax, all nine notebook code cells compile, embedded notebook implementation matches the module, all documentation links to local files resolve, and `git diff --check` is clean. The CNN source, notebook, existing tests, requirements, ignore rules and original ZIPs match the base commit unchanged. The full development test suite was not rerun for this artifact/documentation update.
- The archived split audit reports **8 distinct shared byte-identical image hashes between training and validation**, and 0 between training–test or validation–test. These are distinct shared hashes, not pair counts. Validation overlap and selecting both checkpoint and policy on validation limit interpretation; patient independence cannot be verified.

No training, new inference, source-data download, split change, threshold adjustment or notebook execution was performed in this update. This verifies consistency of the supplied artifacts, not an independent end-to-end reproduction. The original dataset hash and overlap counts are retained as recorded; they were not recomputed from source arrays. The archive contains neither an executed notebook nor a training-time source-code commit. The user's completed Colab run supplies the measurements; Colab UI behavior was not independently observed here.

## Historical development checks (before the Colab run)

The following is the original verification record from before the real results archive was available. These 26 development tests are historical results, not a claim that the dependency-heavy suite was rerun during this documentation update.

- `python -m py_compile src/pneumoguard.py`: passed.
- `python -m pytest -q tests`: **26 tests passed** in the development environment.
- Notebook JSON passed `nbformat.validate`; every code cell compiled as Python.
- The embedded notebook implementation matches the canonical module before its CLI entry point. No Codex-local paths are embedded in the notebook.
- Expected imports were installed and executed: PyTorch, torchvision, medmnist, NumPy, pandas, scikit-learn, Matplotlib, Pillow, ipywidgets, nbformat and nbclient.
- Metrics were compared with independent scikit-learn calculations. Separate scientific review also checked 100 random binary metric cases and 100 random validation policy selections.
- Tests checked majority-class fitting, exact threshold boundaries, deterministic selection and inference, accepted-only denominators, undefined metrics, abstain-all fallback, validation-only policy selection, and test access only after freezing the checkpoint and policy.
- Tests rejected tampered policies and checkpoints before test access.
- Changing test model weights changed fresh demo inference, confirming the demonstration does not use hardcoded predictions.
- A **development-only copy** of all notebook code cells executed successfully with explicitly synthetic test fixtures injected into the loader: training, threshold selection, evaluation, figures, widget construction, demo inference and ZIP export completed with **zero cell errors**. The original deliverable was not modified to use synthetic data. These synthetic outputs are stored outside the repository and are **not benchmark measurements or submission results**.
- Plots were generated and inspected using development fixtures, including zero-coverage and no-error edge cases. Undefined selective error is not plotted as zero.

The development interpreter was Python 3.12.14 with CPU-only PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, medmnist 3.0.2, NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.8.0. Compatible ranges are in `requirements.txt`; completed real runs record their exact runtime in `environment.json`.

## Historical dataset-access block

The official dataset download was attempted using `PneumoniaMNIST(..., download=True, size=28)`. The cloud proxy rejected `zenodo.org` with **403 Forbidden** because that domain was absent from the environment allowlist. No substitute dataset was presented as PneumoniaMNIST.

That block described the initial development environment. It was followed by the user's real Colab run, whose trained checkpoint, full test predictions, measurements and mistakes are now preserved in `results/`. The old block is not the current evaluation status. No synthetic development output was copied into the measured results.

A live Google Colab browser session and its widget rendering/download controls were not independently tested in this update. Historical execution under synthetic fixtures, as described above, is separate from the supplied measured Colab artifacts.

## Remaining actions

1. Review the measured write-up, narration, figures and limitations. No retraining is needed to inspect these results.
2. If the originally executed Colab notebook was saved, retain it alongside the original run. It is not present in the supplied ZIP; do not manufacture executed outputs or rerun training just to fill that gap.
3. Review the pull request and merge the results/documentation update when ready. Confirm the final repository links are viewable without signing in.
4. Complete the human-owned items in the [submission checklist](submission_checklist.md): official detailed brief, eligibility/registration, team details, optional presentation video and manual submission. A live demo should be described as live only if actually run.

This project is a prototype and is not medical advice.
