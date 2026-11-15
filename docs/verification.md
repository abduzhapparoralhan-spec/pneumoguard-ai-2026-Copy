# Verification record

Development date: November 15, 2026. The repository initially contained only a title README. No pre-existing implementation was reused.

## Completed checks

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

## Blocked or untested

The official dataset download was attempted using `PneumoniaMNIST(..., download=True, size=28)`. The cloud proxy rejected `zenodo.org` with **403 Forbidden** because that domain was absent from the environment allowlist. No substitute dataset was presented as PneumoniaMNIST.

Therefore the following remain **not tested on the actual dataset**: measured class counts and image arrays, real-data smoke execution, full CNN training, official unseen-test metrics, benchmark mistakes, and the live demo using a genuinely benchmark-trained checkpoint. No such checkpoint or measured research result is shipped yet.

A live Google Colab browser session and its widget rendering/download controls were not tested. The notebook's Python path was executed under a Jupyter kernel with synthetic fixtures as described above; that does not replace a real-data Colab run.

## Remaining actions

1. Allow the official dataset host in the cloud environment or open the notebook in Google Colab.
2. Run all cells with `SMOKE_TEST = False` without adjusting choices based on test outcomes.
3. Download the results ZIP and the executed notebook. Put full-run artifacts in `results/`.
4. Replace “Not yet evaluated” in the README and write-up with measurements from the generated report; fill narration prompts from that report.
5. Test the public-benchmark demo in Colab, review failures and limitations, then manually publish and submit only after reviewing the official competition brief.

This project is a prototype and is not medical advice.
