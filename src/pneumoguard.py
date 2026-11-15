"""Original educational experiment for PneumoGuard AI, created November 15, 2026.

Only public PneumoniaMNIST images are accepted. No clinical input interface exists.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from medmnist import INFO, PneumoniaMNIST
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DISCLAIMER = ("This project is a prototype and is not medical advice. "
              "Predictions are for research and educational purposes only and "
              "must not be used for clinical decisions.")
LABELS = {0: "Normal", 1: "Pneumonia"}


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    max_epochs: int = 20
    patience: int = 4
    batch_size: int = 64
    learning_rate: float = 0.001
    target_validation_error: float = 0.05
    smoke: bool = False
    data_dir: str = "data"
    output_dir: str = "results"


@dataclass
class ImageSplit:
    name: str
    images: np.ndarray
    labels: np.ndarray
    ids: np.ndarray


@dataclass
class ScoreBatch:
    split: str
    labels: np.ndarray
    scores: np.ndarray
    ids: np.ndarray

    def __post_init__(self):
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.ids = np.asarray(self.ids).reshape(-1)
        if not (len(self.labels) == len(self.scores) == len(self.ids)):
            raise ValueError("Labels, scores and IDs must have equal lengths.")
        if not np.isin(self.labels, [0, 1]).all():
            raise ValueError("Expected original binary benchmark labels.")
        if not np.isfinite(self.scores).all() or ((self.scores < 0) | (self.scores > 1)).any():
            raise ValueError("Scores must be finite and between zero and one.")


@dataclass(frozen=True)
class Policy:
    low: float
    high: float
    target_validation_error: float
    validation_coverage: float
    validation_error: float | None
    selection_split: str = "val"
    abstain_all: bool = False
    rule: str = "score < low: Normal; score >= high: Pneumonia; otherwise: Review"


class SmallCNN(nn.Module):
    """1x28x28 -> 16x14x14 -> 32x7x7 -> 32 -> one logit."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, images):
        return self.layers(images).squeeze(1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(name: str, root: str, limit: int | None = None) -> ImageSplit:
    if name not in {"train", "val", "test"}:
        raise ValueError("Use an official train, val or test split.")
    Path(root).mkdir(parents=True, exist_ok=True)
    try:
        dataset = PneumoniaMNIST(split=name, root=root, download=True, size=28)
    except Exception as exc:
        raise RuntimeError(
            "Could not load official PneumoniaMNIST. In Colab, enable internet and run again. "
            "Alternatively download pneumoniamnist.npz from "
            "https://zenodo.org/records/10519652 into the data folder. "
            "No synthetic substitute has been used."
        ) from exc
    images = np.asarray(dataset.imgs)
    labels = np.asarray(dataset.labels).reshape(-1).astype(np.int64)
    if images.shape != (len(labels), 28, 28) or images.dtype != np.uint8:
        raise ValueError(f"Unexpected image shape/type: {images.shape}, {images.dtype}")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Dataset must preserve labels 0 Normal and 1 Pneumonia.")
    expected = INFO["pneumoniamnist"]["n_samples"][name]
    if len(labels) != expected:
        raise ValueError("Dataset split size does not match official metadata.")
    indices = np.arange(len(labels))
    if limit is not None and limit < len(indices):
        # Fixed sampling within this split only; never pool or re-split the benchmark.
        indices = np.sort(np.random.default_rng(2026).choice(indices, limit, replace=False))
    return ImageSplit(name, images[indices], labels[indices],
                      np.array([f"{name}:{i}" for i in indices]))


def image_tensor(images: np.ndarray) -> torch.Tensor:
    # Fixed scaling requires no fitted statistics and is identical for all splits.
    return torch.from_numpy(np.asarray(images).copy()).float().unsqueeze(1) / 255.0


def make_loader(split: ImageSplit, batch_size: int, shuffle: bool, seed: int = 2026):
    dataset = TensorDataset(image_tensor(split.images), torch.from_numpy(split.labels).float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def majority_class(training_labels: np.ndarray) -> int:
    labels = np.asarray(training_labels).reshape(-1)
    if len(labels) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Need nonempty binary training labels.")
    # Ties select Normal deterministically. The majority is learned from training only.
    return int(np.argmax(np.bincount(labels.astype(int), minlength=2)))


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    y, p = np.asarray(labels).reshape(-1), np.asarray(predictions).reshape(-1)
    if len(y) != len(p) or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Expected equal-length binary labels and predictions.")
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())
    ratio = lambda a, b: float(a / b) if b else None
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "n": len(y), "accuracy": ratio(tp + tn, len(y)),
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None else None,
        "precision_pneumonia": ratio(tp, tp + fp), "recall_pneumonia": recall,
        "f1_pneumonia": ratio(2 * tp, 2 * tp + fp + fn),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_negatives": fn, "false_positives": fp, "errors": fp + fn,
    }


def apply_policy(scores: np.ndarray, policy: Policy) -> np.ndarray:
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite and between zero and one.")
    if not 0 <= policy.low <= 0.5 <= policy.high <= 1:
        raise ValueError("Thresholds must bracket the fixed forced-choice boundary 0.5.")
    decisions = np.full(scores.shape, -1, dtype=np.int64)
    if not policy.abstain_all:
        decisions[scores < policy.low] = 0
        decisions[scores >= policy.high] = 1
    return decisions


def select_policy(validation: ScoreBatch, target_error: float = 0.05):
    """Validation-only deterministic grid. No test inputs or metrics are accepted."""
    if validation.split != "val" or not all(str(i).startswith("val:") for i in validation.ids):
        raise ValueError("Policy selection requires validation predictions and validation IDs only.")
    if not len(validation.labels) or not 0 <= target_error <= 1:
        raise ValueError("Need nonempty validation data and a target in [0, 1].")
    rows, eligible = [], []
    for low in np.round(np.arange(0, 0.501, 0.05), 2):
        for high in np.round(np.arange(0.5, 1.001, 0.05), 2):
            policy = Policy(float(low), float(high), target_error, 0, None)
            pred = apply_policy(validation.scores, policy)
            accepted = pred != -1
            n = int(accepted.sum())
            errors = int((pred[accepted] != validation.labels[accepted]).sum())
            risk = errors / n if n else None
            row = {"low": float(low), "high": float(high), "accepted": n,
                   "coverage": n / len(pred), "automatic_errors": errors,
                   "automatic_error_rate": risk}
            rows.append(row)
            if n and risk <= target_error:
                eligible.append(row)
    if not eligible:
        selected = Policy(0.0, 1.0, target_error, 0.0, None, abstain_all=True)
    else:
        best = min(eligible, key=lambda r: (-r["accepted"], r["automatic_error_rate"], r["low"], r["high"]))
        selected = Policy(best["low"], best["high"], target_error,
                          best["coverage"], best["automatic_error_rate"])
    return selected, pd.DataFrame(rows)


def selective_metrics(batch: ScoreBatch, policy: Policy):
    decisions = apply_policy(batch.scores, policy)
    accepted = decisions != -1
    metrics = classification_metrics(batch.labels[accepted], decisions[accepted])
    total = len(decisions)
    matrix = [[int(((batch.labels == label) & (decisions == dec)).sum())
               for dec in [0, 1, -1]] for label in [0, 1]]
    return {
        "n_total": total, "n_automatic": int(accepted.sum()),
        "n_review": int((~accepted).sum()),
        "coverage": float(accepted.mean()) if total else None,
        "review_rate": float((~accepted).mean()) if total else None,
        "automatic_error_rate": metrics["errors"] / metrics["n"] if metrics["n"] else None,
        "automatic_errors_per_all_cases": metrics["errors"] / total if total else None,
        "abstained_normal": matrix[0][2], "abstained_pneumonia": matrix[1][2],
        "reference_by_decision_matrix": matrix,
        "automatic_only": metrics,
    }


@torch.inference_mode()
def predict_scores(model: SmallCNN, split: ImageSplit, device: torch.device, batch_size=128):
    model.eval()
    scores = []
    for images, _ in make_loader(split, batch_size, False):
        scores.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
    return ScoreBatch(split.name, split.labels, np.concatenate(scores), split.ids)


def split_description(split: ImageSplit):
    return {"n": len(split.labels), "shape": list(split.images.shape),
            "class_counts": {LABELS[k]: int((split.labels == k).sum()) for k in [0, 1]},
            "ids_sha256": hashlib.sha256("\n".join(split.ids).encode()).hexdigest(),
            "images_sha256": hashlib.sha256(split.images.tobytes()).hexdigest()}


@dataclass
class Experiment:
    config: Config
    model: SmallCNN
    device: torch.device
    train: ImageSplit
    val: ImageSplit
    policy: Policy
    majority: int
    policy_sha256: str
    history: pd.DataFrame
    validation_grid: pd.DataFrame
    best_epoch: int
    test: ImageSplit | None = None
    test_batch: ScoreBatch | None = None
    metrics: dict | None = None


def train_and_freeze(config: Config) -> Experiment:
    if config.max_epochs < 1 or config.patience < 1 or config.batch_size < 1:
        raise ValueError("Epochs, patience and batch size must be positive.")
    out = Path(config.output_dir)
    if (out / "metrics.json").exists():
        raise FileExistsError("A completed run exists here. Use a new output folder to preserve it.")
    out.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split("train", config.data_dir, 256 if config.smoke else None)
    val = load_split("val", config.data_dir, 128 if config.smoke else None)
    assert not set(train.ids) & set(val.ids)
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = make_loader(train, config.batch_size, True, config.seed)
    val_loader = make_loader(val, config.batch_size, False)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    max_epochs = 2 if config.smoke else config.max_epochs
    print(f"Training on {device}: {len(train.labels)} train / {len(val.labels)} validation images.")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(labels)
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += float(loss_fn(logits, labels)) * len(labels)
                correct += int(((logits >= 0).long() == labels.long()).sum())
        val_loss /= len(val.labels)
        row = {"epoch": epoch, "train_loss": train_loss / len(train.labels),
               "val_loss": val_loss, "val_accuracy": correct / len(val.labels)}
        history.append(row)
        print(f"Epoch {epoch:02d} | train loss {row['train_loss']:.4f} | "
              f"val loss {val_loss:.4f} | val accuracy {row['val_accuracy']:.3f}")
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       out / "best_model.pt")
        else:
            stale += 1
        if stale >= config.patience:
            print("Early stopping: no validation loss improvement.")
            break
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device, weights_only=True))
    validation = predict_scores(model, val, device)
    policy, grid = select_policy(validation, config.target_validation_error)
    write_json(out / "policy.json", asdict(policy))
    write_json(out / "config.json", asdict(config))
    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    grid.to_csv(out / "validation_policy_grid.csv", index=False)
    pd.DataFrame({"id": validation.ids, "reference_label": validation.labels,
                  "score": validation.scores}).to_csv(out / "validation_predictions.csv", index=False)
    write_json(out / "training_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_splits": ["train", "val"], "test_used_for_selection": False,
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": file_sha256(out / "best_model.pt"),
        "policy_sha256": file_sha256(out / "policy.json"),
    })
    print(f"Policy frozen before test access: low={policy.low}, high={policy.high}, "
          f"validation coverage={policy.validation_coverage:.3f}")
    return Experiment(config, model, device, train, val, policy, majority_class(train.labels),
                      file_sha256(out / "policy.json"), pd.DataFrame(history), grid, best_epoch)


def evaluate_test(experiment: Experiment) -> dict:
    """Open the official test split only after weights and policy have been frozen."""
    if experiment.metrics is not None:
        return experiment.metrics
    out = Path(experiment.config.output_dir)
    if file_sha256(out / "policy.json") != experiment.policy_sha256:
        raise ValueError("Frozen policy changed before final evaluation.")
    stored = Policy(**json.loads((out / "policy.json").read_text()))
    if stored != experiment.policy or stored.selection_split != "val":
        raise ValueError("Policy provenance check failed.")
    training_manifest = json.loads((out / "training_manifest.json").read_text())
    if file_sha256(out / "best_model.pt") != training_manifest["checkpoint_sha256"]:
        raise ValueError("Frozen model checkpoint changed before final evaluation.")
    # Reload the frozen checkpoint so the evaluated model cannot be an accidental later state.
    experiment.model.load_state_dict(torch.load(out / "best_model.pt",
                                               map_location=experiment.device, weights_only=True))
    test = load_split("test", experiment.config.data_dir, 128 if experiment.config.smoke else None)
    assert not (set(test.ids) & (set(experiment.train.ids) | set(experiment.val.ids)))
    batch = predict_scores(experiment.model, test, experiment.device)
    forced = (batch.scores >= 0.5).astype(np.int64)
    baseline = np.full(len(test.labels), experiment.majority)
    decisions = apply_policy(batch.scores, stored)
    metrics = {"status": "SMOKE TEST ONLY" if experiment.config.smoke else "Measured full test evaluation",
               "majority_class": LABELS[experiment.majority],
               "baseline": classification_metrics(test.labels, baseline),
               "cnn_forced": classification_metrics(test.labels, forced),
               "cnn_selective": selective_metrics(batch, stored)}
    pd.DataFrame({"id": batch.ids, "reference_label": batch.labels,
                  "pneumonia_score": batch.scores, "baseline_prediction": baseline,
                  "forced_prediction": forced, "policy_decision": decisions,
                  "forced_correct": forced == batch.labels,
                  "automatic": decisions != -1,
                  "automatic_correct": [bool(p == y) if p != -1 else None
                                        for p, y in zip(decisions, batch.labels)]}).to_csv(
                                            out / "test_predictions.csv", index=False)
    splits = {s.name: split_description(s) for s in [experiment.train, experiment.val, test]}
    # A content audit is descriptive only; it never changes official test membership.
    hashes = {s.name: {hashlib.sha256(im.tobytes()).hexdigest() for im in s.images}
              for s in [experiment.train, experiment.val, test]}
    overlaps = {f"{a}-{b}": len(hashes[a] & hashes[b])
                for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
    write_json(out / "split_manifest.json", {"splits": splits, "identical_image_hash_overlap": overlaps,
              "patient_independence": "Cannot independently verify: patient IDs are not supplied."})
    versions = {}
    for package in ["torch", "torchvision", "medmnist", "numpy", "pandas", "scikit-learn",
                    "matplotlib", "ipywidgets", "nbformat", "nbclient"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    write_json(out / "environment.json", {"python": platform.python_version(),
               "platform": platform.platform(), "device": str(experiment.device),
               "packages": versions, "seed": experiment.config.seed,
               "dataset_npz_sha256": file_sha256(Path(experiment.config.data_dir) / "pneumoniamnist.npz")})
    write_json(out / "metrics.json", metrics)
    experiment.test, experiment.test_batch, experiment.metrics = test, batch, metrics
    write_report(experiment)
    return metrics


def comparison_table(experiment: Experiment) -> pd.DataFrame:
    if experiment.metrics is None:
        raise ValueError("Run final evaluation first.")
    m = experiment.metrics
    rows = []
    for name, item, coverage in [
        ("Majority baseline (all test)", m["baseline"], 1.0),
        ("CNN forced choice (all test)", m["cnn_forced"], 1.0),
        ("CNN abstention (automatic subset)", m["cnn_selective"]["automatic_only"],
         m["cnn_selective"]["coverage"]),
    ]:
        rows.append({"method": name, "evaluated_n": item["n"], "coverage": coverage,
                     **{k: item[k] for k in ["accuracy", "balanced_accuracy", "precision_pneumonia",
                                              "recall_pneumonia", "f1_pneumonia", "false_negatives",
                                              "false_positives", "errors"]}})
    return pd.DataFrame(rows)


def _fmt(value, percent=True):
    if value is None:
        return "N/A"
    return f"{value:.2%}" if percent else str(value)


def write_report(experiment: Experiment):
    out = Path(experiment.config.output_dir)
    m = experiment.metrics
    s, f = m["cnn_selective"], m["cnn_forced"]
    table = comparison_table(experiment)
    table.to_csv(out / "comparison.csv", index=False)
    columns = ["Method", "N", "Coverage", "Accuracy", "Balanced acc.", "Precision", "Recall", "F1", "FN", "FP"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in table.to_dict("records"):
        lines.append("| " + " | ".join([r["method"], str(r["evaluated_n"]), _fmt(r["coverage"])] +
                     [_fmt(r[k]) if pd.notna(r[k]) else "N/A" for k in ["accuracy", "balanced_accuracy",
                      "precision_pneumonia", "recall_pneumonia", "f1_pneumonia"]] +
                     [str(r["false_negatives"]), str(r["false_positives"])]) + " |")
    risk = s["automatic_error_rate"]
    forced_risk = f["errors"] / f["n"]
    if risk is None:
        finding = "The policy abstained on every test case; automatic error rate is undefined."
    elif risk < forced_risk:
        finding = (f"Selective error decreased from {_fmt(forced_risk)} to {_fmt(risk)}, while "
                   f"{s['n_review']} of {s['n_total']} images required review. This comparison uses different coverage.")
    elif risk > forced_risk:
        finding = (f"Selective error increased from {_fmt(forced_risk)} to {_fmt(risk)}. "
                   "The validation-selected policy did not improve this test metric; no test-based retuning was performed.")
    else:
        finding = "Selective error did not improve over forced choice on this test set."
    text = f"""# Measured experiment report

**{m['status']}**. One seed ({experiment.config.seed}); best validation-loss checkpoint: epoch {experiment.best_epoch}.

{DISCLAIMER}

Research question: Can abstention reduce incorrect automatic decisions, and at what manual-review workload?

{chr(10).join(lines)}

Precision, recall and F1 refer to Pneumonia. Undefined denominators are N/A.
The abstention row evaluates only accepted cases; it does not measure whole-test accuracy or whole-test recall.
The baseline and forced CNN evaluate the entire test set. Do not rank these accuracies without their coverage.

## Frozen validation policy

Normal if score < {experiment.policy.low}; Pneumonia if score >= {experiment.policy.high}; otherwise review.
Abstain-all fallback: {experiment.policy.abstain_all}.
Selected on validation only by maximum coverage subject to empirical error <= {experiment.config.target_validation_error:.0%}.
Ties: lower error, then ascending low/high thresholds. Fixed grid spacing: 0.05.
Validation coverage: {_fmt(experiment.policy.validation_coverage)}; validation automatic error: {_fmt(experiment.policy.validation_error)}.
The target is a design choice, not a medical safety guarantee. Scores are uncalibrated sigmoid outputs.
The same validation split chooses the checkpoint and policy; its performance can be optimistic.

## Test workload and errors

- Automatic decisions: {s['n_automatic']}/{s['n_total']} ({_fmt(s['coverage'])}).
- Manual review: {s['n_review']}/{s['n_total']} ({_fmt(s['review_rate'])}).
- Automatic errors / accepted images: {s['automatic_only']['errors']}/{s['n_automatic']} ({_fmt(risk)}).
- Automatic errors / all test images: {s['automatic_only']['errors']}/{s['n_total']} ({_fmt(s['automatic_errors_per_all_cases'])}).
- Automatic false negatives: {s['automatic_only']['false_negatives']}; automatic false positives: {s['automatic_only']['false_positives']}.
- Abstained benchmark Normal: {s['abstained_normal']}; abstained benchmark Pneumonia: {s['abstained_pneumonia']}.

{finding}

Abstained examples are not counted as correct, false negative, or false positive automatic decisions.
No assumption is made about how a human would classify reviewed images.
Fewer automatic errors alone can result from making fewer decisions.

## Confusion matrices

Rows: benchmark Normal, Pneumonia. Columns: predicted Normal, Pneumonia.

- Baseline: `{m['baseline']['confusion_matrix']}`
- Forced CNN: `{f['confusion_matrix']}`
- Accepted-only CNN: `{s['automatic_only']['confusion_matrix']}`

For the full workload matrix, columns are Normal, Pneumonia, Review:
`{s['reference_by_decision_matrix']}`.

## Interpretation and limits

Overall accuracy can favor the majority class; balanced accuracy averages recall for both labels.
The majority baseline cannot distinguish image content. Small images lose clinically important detail.
This single-seed, single-benchmark evaluation is descriptive, not evidence of clinical effectiveness.
The test set is small, and selecting thresholds by an empirical validation target provides no guaranteed future error bound.
Dataset annotations are reference labels, not independent clinical confirmation.
Pediatric images from one source do not establish performance on adults, other sites or devices.
Patient IDs are unavailable, so we cannot independently verify patient-level separation.
See split_manifest.json for exact image overlap, environment.json for tested versions, and test_predictions.csv for every output.
Failure examples and a score-based explanation are generated by create_figures().
"""
    (out / "report.md").write_text(text, encoding="utf-8")


def create_figures(experiment: Experiment):
    import matplotlib.pyplot as plt
    if experiment.metrics is None or experiment.test is None:
        raise ValueError("Evaluate before plotting.")
    out = Path(experiment.config.output_dir)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    m, batch = experiment.metrics, experiment.test_batch
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    matrices = [m["baseline"]["confusion_matrix"], m["cnn_forced"]["confusion_matrix"],
                m["cnn_selective"]["automatic_only"]["confusion_matrix"],
                m["cnn_selective"]["reference_by_decision_matrix"]]
    titles = ["Baseline: all test", "CNN: all test", "CNN: accepted only", "Policy: full workload"]
    for ax, matrix, title in zip(axes, matrices, titles):
        arr = np.asarray(matrix)
        ax.imshow(arr, cmap="Blues", vmin=0)
        for (r, c), val in np.ndenumerate(arr):
            ax.text(c, r, str(val), ha="center", va="center", color="black",
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_xticks(range(arr.shape[1]), ["Normal", "Pneumonia", "Review"][:arr.shape[1]], rotation=20)
        ax.set_yticks([0, 1], ["Normal", "Pneumonia"])
        ax.set(xlabel="Model decision", ylabel="Benchmark reference", title=title)
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    h = experiment.history
    axes[0].plot(h.epoch, h.train_loss, label="Training loss")
    axes[0].plot(h.epoch, h.val_loss, label="Validation loss")
    axes[0].axvline(experiment.best_epoch, color="#d18a00", linestyle="--", label="Selected checkpoint")
    axes[0].set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Training history")
    axes[0].legend()
    grid = experiment.validation_grid.dropna(subset=["automatic_error_rate"])
    axes[1].scatter(grid.coverage * 100, grid.automatic_error_rate * 100, s=15, alpha=.35, label="Validation grid")
    if experiment.policy.validation_error is not None:
        axes[1].scatter([experiment.policy.validation_coverage * 100],
                        [experiment.policy.validation_error * 100], marker="*", s=160, label="Selected on validation")
    else:
        axes[1].text(.05, .9, "Selected: no automatic decisions; error undefined",
                     transform=axes[1].transAxes, fontsize=8)
    axes[1].axhline(experiment.config.target_validation_error * 100, color="#d18a00", linestyle="--", label="Experimental target")
    axes[1].set(xlabel="Automatic coverage (%)", ylabel="Errors / automatic decisions (%)",
                title="Validation coverage–error trade-off")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "training_and_validation_tradeoff.png", dpi=150)
    plt.close(fig)

    forced = (batch.scores >= .5).astype(int)
    policy_decisions = apply_policy(batch.scores, experiment.policy)
    mistakes = np.flatnonzero(forced != batch.labels)
    # First five benchmark-index mistakes; no cherry-picking by apparent clinical content.
    selected = list(mistakes[:5])
    if len(selected) < 5:
        uncertain = np.argsort(np.abs(batch.scores - .5), kind="stable")
        selected += [int(i) for i in uncertain if int(i) not in selected][:5-len(selected)]
    rows = []
    fig, axes = plt.subplots(1, len(selected), figsize=(3.1 * len(selected), 3.5), squeeze=False)
    for ax, i in zip(axes[0], selected):
        decision = "Review" if policy_decisions[i] == -1 else LABELS[int(policy_decisions[i])]
        kind = ("False negative" if forced[i] == 0 else "False positive") if forced[i] != batch.labels[i] else "Closest to boundary (correct)"
        rows.append({"id": str(batch.ids[i]), "kind": kind, "reference_label": LABELS[int(batch.labels[i])],
                     "forced_prediction": LABELS[int(forced[i])], "score": float(batch.scores[i]), "policy_decision": decision})
        ax.imshow(experiment.test.images[i], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
        ax.axis("off")
        ax.set_title(f"{batch.ids[i]} | {kind}\nReference: {LABELS[int(batch.labels[i])]}\n"
                     f"Score: {batch.scores[i]:.3f} | {decision}", fontsize=9)
    fig.suptitle("Actual benchmark outputs — 28×28 images enlarged for display")
    fig.tight_layout()
    fig.savefig(figures / "failure_examples.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "failure_examples.csv", index=False)
    first = rows[0]
    heading = "One observed failure" if len(mistakes) else "Closest-to-boundary example (no forced-choice errors found)"
    explanation = (f"\n\n## {heading}\n\n{len(mistakes)} forced-choice mistakes were found. "
        "The figure shows the first five mistakes in test-index order, supplemented by correct examples closest to the boundary only if needed. "
        f"For {first['id']}, the reference label is {first['reference_label']} and the "
        f"CNN forced prediction is {first['forced_prediction']} (score {first['score']:.4f}). "
        f"The frozen policy returns {first['policy_decision']}. "
        "This illustrates the behavior of the score and threshold rule. We cannot determine the clinical reason "
        "for a disagreement from a low-resolution image or infer a diagnosis.\n")
    with (out / "report.md").open("a", encoding="utf-8") as stream:
        stream.write(explanation)
    return pd.DataFrame(rows)


def infer_test_image(experiment: Experiment, index: int) -> dict:
    """Fresh model inference, restricted to an evaluated public benchmark index."""
    if experiment.test is None:
        raise ValueError("Run final evaluation before using the demo.")
    if not isinstance(index, (int, np.integer)) or not 0 <= index < len(experiment.test.labels):
        raise ValueError("Select an existing public test-image index.")
    image = experiment.test.images[index:index + 1]
    experiment.model.eval()
    with torch.inference_mode():
        score = float(torch.sigmoid(experiment.model(image_tensor(image).to(experiment.device)))[0].cpu())
    forced = int(score >= .5)
    decision = int(apply_policy(np.array([score]), experiment.policy)[0])
    label = int(experiment.test.labels[index])
    return {"id": str(experiment.test.ids[index]), "pneumonia_score": score,
            "normal_score": 1 - score, "forced_prediction": LABELS[forced],
            "policy_status": "Uncertain — manual review required" if decision == -1 else f"Model favors {LABELS[decision]}",
            "benchmark_reference": LABELS[label], "forced_matches_reference": forced == label,
            "automatic_matches_reference": None if decision == -1 else decision == label}


def show_demo(experiment: Experiment):
    """Simple Colab-compatible widgets. No uploads, private scans or external service."""
    import html
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import HTML, clear_output, display

    if experiment.test is None:
        raise ValueError("Run final evaluation first.")
    display(HTML(f'<div style="background:#fff;color:#10213b;padding:18px;border:1px solid #dce5ef;">'
                 f'<h2>PneumoGuard AI · Public benchmark explorer</h2><b>{DISCLAIMER}</b>'
                 '<p>Choose a public test image, then run the trained CNN. Scores are uncalibrated model outputs. '
                 'Reference labels are benchmark annotations.</p></div>'))
    selector = widgets.BoundedIntText(value=0, min=0, max=len(experiment.test.labels)-1,
                                     description="Test index:", layout=widgets.Layout(width="260px"))
    button = widgets.Button(description="Run model", button_style="primary", icon="play")
    output = widgets.Output()

    def run(_):
        with output:
            clear_output(wait=True)
            result = infer_test_image(experiment, int(selector.value))
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(experiment.test.images[selector.value], cmap="gray", interpolation="nearest", vmin=0, vmax=255)
            ax.set_title(f"{result['id']} · original resolution 28×28")
            ax.axis("off")
            plt.show()
            color = "#996000" if result["automatic_matches_reference"] is None else "#124ea0"
            automatic = ("No automatic decision" if result["automatic_matches_reference"] is None else
                         "Matches reference" if result["automatic_matches_reference"] else "Differs from reference")
            display(HTML(f'<div style="background:#fff;color:#10213b;padding:12px;">'
                f'<h3 style="color:{color}">{html.escape(result["policy_status"])}</h3>'
                f'<p>Forced CNN category: <b>{result["forced_prediction"]}</b><br>'
                f'Pneumonia score: {result["pneumonia_score"]:.4f} · Normal score: {result["normal_score"]:.4f}<br>'
                f'Benchmark reference: <b>{result["benchmark_reference"]}</b><br>'
                f'Forced prediction matches reference: {result["forced_matches_reference"]}<br>'
                f'Policy result: {automatic}</p><p>{DISCLAIMER}</p></div>'))
    button.on_click(run)
    display(widgets.HBox([selector, button]), output)
    run(None)
    return {"selector": selector, "button": button, "output": output}


def run_experiment(config: Config) -> Experiment:
    experiment = train_and_freeze(config)
    evaluate_test(experiment)
    create_figures(experiment)
    return experiment


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate on public PneumoniaMNIST only.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default="results")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    cfg = Config(smoke=args.smoke, output_dir=args.output, data_dir=args.data)
    exp = run_experiment(cfg)
    print(comparison_table(exp).to_string(index=False))
    print(DISCLAIMER)
