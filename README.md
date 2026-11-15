# PneumoGuard AI

**Can a classifier make fewer incorrect automatic decisions by referring uncertain public benchmark images for review?**

Veritas AI International High School Hackathon 2026 · AI + Health

> **This project is a prototype and is not medical advice. Predictions are for research and educational purposes only and must not be used for clinical decisions.**

PneumoGuard AI is an educational research prototype for **28×28 public pediatric chest X-ray benchmark images**. Its intended user is a hypothetical healthcare worker considering which images deserve additional review. The application accepts only public test-set indices. It does not accept private scans, identify patients, diagnose disease, or recommend treatment.

## Start in Google Colab

1. Download this repository and locate [notebooks/PneumoGuard_AI.ipynb](notebooks/PneumoGuard_AI.ipynb).
2. Open [Google Colab](https://colab.research.google.com/) → **File → Upload notebook** → select that file. Save a copy to Drive.
3. Leave `SMOKE_TEST = False` and choose **Runtime → Run all**. CPU is supported. No API key or repository clone is needed: the notebook includes its own implementation.
4. Wait for training, evaluation, figures and the benchmark explorer. Click **Run model** for a selected public test index.
5. Click **Download results ZIP**, then **File → Download → Download .ipynb**. Extract the full-run results into `results/` and update the submission text from the generated report.

Every notebook run creates a separate timestamped folder in `results/colab_run/`. Optional smoke mode uses smaller, separate split subsets and two epochs; it is **not a reportable full evaluation**. Download outputs before Colab disconnects. Static notebook previews may show saved results but require execution to restore interactive widgets.

Simple Russian click-by-click instructions: [student guide](docs/student_guide_ru.md).

## Research question and contribution

**Can an uncertainty-aware image classifier reduce incorrect automatic decisions compared with a standard forced-choice classifier, and what is the trade-off in manual review workload?**

The experiment connects a small CNN, a validation-selected abstention policy, accepted-case errors and a measurable review workload. This is the project's original experimental contribution beyond a basic image classifier. It does not claim invention of selective classification or a clinically useful tool. Rejecting more cases can reduce automatic errors while increasing human work; the study measures both. It does not assume human reviewers always make correct decisions.

## Dataset and separation

[PneumoniaMNIST](https://github.com/MedMNIST/MedMNIST) is loaded with the official `medmnist` package from the [official Zenodo distribution](https://zenodo.org/records/10519652).

| Property | Setting |
|---|---|
| Images | 28×28 grayscale; pediatric source; enlarged only for display |
| Labels | Original 0 = Normal; 1 = Pneumonia |
| Official splits | 4,708 training; 524 validation; 624 test images |
| Training | Only training images update model weights |
| Validation | Checkpoint selection and abstention-threshold selection |
| Test | Final evaluation after weights and thresholds are frozen |
| Class counts | Generated from loaded arrays in `split_manifest.json`; not inferred from overall accuracy |
| Dataset license | PneumoniaMNIST CC BY 4.0; source Mendeley dataset CC BY 4.0 |

The official archive contains all three splits, but the training stage accesses only training and validation arrays. It neither pools nor re-splits the benchmark. Fixed scaling divides pixels by 255 and fits no preprocessing on test data. Split IDs, image hashes, cross-split byte-identical image overlaps, archive SHA-256 and package versions are recorded. No claim of verified patient-level independence is made: patient IDs are not supplied. The source annotations are benchmark reference labels, not independent clinical confirmation.

The source is Kermany et al.'s pediatric chest X-ray dataset. Low resolution, restricted source population, class imbalance, possible dataset artifacts and limited sample size constrain generalization. Do not apply conclusions to adults, other hospitals, devices or private scans.

## Model and baseline

The baseline always predicts the most common **training** label. Ties deterministically choose Normal. On an imbalanced dataset, a majority rule may have substantial accuracy while failing completely for the minority class; balanced accuracy averages the two class recalls.

The original PyTorch CNN has **55,041 trainable parameters**:

```text
1 × 28 × 28 grayscale input, pixels / 255
  → Conv 3×3, 16 channels → ReLU → MaxPool 2×2
  → Conv 3×3, 32 channels → ReLU → MaxPool 2×2
  → Flatten (32×7×7) → Linear 32 → ReLU → Linear 1
  → Sigmoid score
```

Training uses unweighted `BCEWithLogitsLoss`, Adam (learning rate 0.001), batch size 64, at most 20 epochs and early-stopping patience 4. The checkpoint with the lowest validation loss is restored. Seed 2026, deterministic PyTorch operations where supported, CPU/GPU autodetection and fixed split membership support reproduction. No pretrained weights or external model API are used. Exact numerical agreement across different hardware/library versions is not guaranteed; each run records its environment.

## Validation-selected abstention

Forced choice favors Pneumonia if `score >= 0.5`, otherwise Normal. The extra policy can produce:

- **Model favors Normal:** `score < low`.
- **Model favors Pneumonia:** `score >= high`.
- **Uncertain — manual review required:** all other scores.

The fixed grid uses `low ∈ {0, 0.05, …, 0.5}` and `high ∈ {0.5, 0.55, …, 1}`. Using validation predictions only, select maximum coverage subject to empirical errors / accepted images ≤ 5%. Ties prefer lower error and then ascending low/high thresholds. If no nonempty candidate meets the target, abstain on all cases and report automatic error as undefined.

The 5% target is an experimental design choice, **not a safety guarantee or calibrated risk limit**. Scores are uncalibrated sigmoid outputs. Checkpoint and policy reuse the validation split, so validation performance may be optimistic. Thresholds and checkpoint are saved and hashed before opening the test split. Evaluation verifies these hashes and reloads the frozen checkpoint. A policy that fails to improve test performance is retained and reported honestly.

## Evaluation status

**Not yet evaluated.** At initial verification, the cloud environment blocked the official Zenodo download. No public-dataset training, test performance or clinical effectiveness is claimed at this stage. Run the notebook in Colab or enable the official dataset host to populate the measured outputs.

No accuracy percentages or test confusion matrices are placeholders. After a successful full run, [results/report.md](results/report.md) is the authoritative measured report. It includes:

- Majority baseline and forced CNN: accuracy, balanced accuracy, Pneumonia precision/recall/F1, confusion matrix, FN and FP on **all test images**.
- Abstaining CNN: the same classification metrics on **accepted images only**, with evaluated sample count and coverage.
- Full-test coverage, review rate, automatic errors / accepted images, automatic errors / all test images, automatic FN/FP and reviewed counts by reference class.
- A complete 2×3 matrix (reference × Normal/Pneumonia/Review) and accepted-only 2×2 matrix.
- Training curves, the validation coverage–error grid, and at least five actual mistakes when available. Fewer mistakes are supplemented by correct examples closest to the boundary, explicitly labeled.

Accepted-only accuracy and recall are not complete-test accuracy and recall. Reviewed examples are neither correct automatic classifications nor automatic FN/FP. Undefined denominators are **N/A**, not zero or perfect performance. Report the coverage whenever presenting selective metrics. A reduced automatic error burden can partly reflect making fewer decisions.

Failure analysis explains observed labels, scores and policy decisions. It does not invent medical explanations for disagreement from 28×28 images. One seed on one small benchmark cannot demonstrate clinical effectiveness or guarantee future error rates. No calibration or external-site evaluation is claimed.

## Reproduce locally (optional)

The Colab instructions above require no programming knowledge. For a Python user, from this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pneumoguard --smoke --output results/smoke
python -m src.pneumoguard --output results/reproduction
```

On Windows, activate with `.venv\Scripts\activate`. CPU-only PyTorch wheels can be installed from the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before the requirements. Each completed CLI output directory is protected from accidental overwrite: use a new directory for another run. Do not use repeated test evaluations to tune the model. The command-line smoke and full runs use separate directories and never mix splits.

Load a documented checkpoint with the matching architecture:

```python
import json
import torch
from src.pneumoguard import SmallCNN, Policy

model = SmallCNN()
model.load_state_dict(torch.load("results/best_model.pt", map_location="cpu", weights_only=True))
model.eval()
with open("results/policy.json", encoding="utf-8") as stream:
    policy = Policy(**json.load(stream))
```

Only load checkpoints from a trusted source. The notebook normally trains and restores its own checkpoint; no checkpoint is required to start it.

## Files and measured artifacts

```text
notebooks/PneumoGuard_AI.ipynb    self-contained Colab experiment
src/pneumoguard.py               canonical original implementation
tests/                          scientific logic and notebook checks
docs/one_page_writeup.md         concise competition write-up
docs/demo_script.md              two-minute English demo narration
docs/model_explained.md          architecture and FN/FP explanations
docs/student_guide_ru.md         noncoding student's instructions
docs/submission_checklist.md     manual submission steps
docs/sources.md                  citations, licenses and tools
results/                        measured outputs after real evaluation
```

A completed run writes `best_model.pt`, `policy.json`, `config.json`, `training_manifest.json`, `training_history.csv`, `validation_predictions.csv`, `validation_policy_grid.csv`, `test_predictions.csv`, `metrics.json`, `comparison.csv`, `split_manifest.json`, `environment.json`, `report.md`, `failure_examples.csv` and `figures/*.png`. Downloaded data, virtual environments and smoke outputs should not be uploaded to GitHub.

## Competition and originality

The [official submission page](https://highschoolhackathon.org/briefs) was read during development. Its linked Google Drive project-brief PDF did not expose readable content in this environment; the supplied task brief therefore also governed implementation. Read the complete official PDF before submitting. The public page requires a viewable code/notebook link, write-up, track, team details and credits; it recommends a demo video up to two minutes. Verify current submission requirements and access to every link yourself.

Original code and documentation were created in this session on November 15, 2026. The initial repository contained only a title README. No existing project's implementation or pretrained model was reused. The student must understand and accurately disclose all contributions, verify eligibility and provide their own team details. Nothing is published, submitted or awarded automatically.

## Credits

- Yang et al. (2023), **MedMNIST v2**, Scientific Data 10, 41. [Paper](https://doi.org/10.1038/s41597-022-01721-8), [official package/code](https://github.com/MedMNIST/MedMNIST), [official data](https://zenodo.org/records/10519652).
- Yang, Shi and Ni (2021), **MedMNIST Classification Decathlon**, ISBI, pp. 191–195. [Paper](https://doi.org/10.1109/ISBI48211.2021.9434062).
- Kermany, Zhang and Goldbaum (2018), **Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification**, Mendeley Data v2. [Source dataset](https://doi.org/10.17632/rscbjbr9sj.2), CC BY 4.0.
- Kermany et al. (2018), **Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning**, Cell. [Paper](https://doi.org/10.1016/j.cell.2018.02.010).
- Python; [PyTorch](https://pytorch.org/), [torchvision](https://pytorch.org/vision/stable/), [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/), [Pillow](https://python-pillow.org/), [ipywidgets](https://ipywidgets.readthedocs.io/), [Jupyter/nbformat/nbclient](https://jupyter.org/), [pytest](https://pytest.org/), Google Colab and GitHub. Runtime versions are recorded after evaluation.
- **ChatGPT and OpenAI Codex** assisted with original implementation, experiment design, testing and documentation. This assistance must be disclosed. No tutorial implementation was copied. See [sources and licenses](docs/sources.md).

**This project is a prototype and is not medical advice.**
