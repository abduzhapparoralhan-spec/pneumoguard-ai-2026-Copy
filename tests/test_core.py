"""Scientific logic checks. All arrays here are synthetic test fixtures, never results."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn import metrics as sk_metrics

from src import pneumoguard as pg


def score_batch(split, labels, scores, id_split=None):
    prefix = id_split or split
    return pg.ScoreBatch(split, np.array(labels), np.array(scores),
                         np.array([f"{prefix}:{i}" for i in range(len(labels))]))


def test_metrics_match_independent_sklearn_calculations():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    p = np.array([0, 1, 1, 0, 1, 1, 1, 1])
    measured = pg.classification_metrics(y, p)
    assert measured["accuracy"] == pytest.approx(sk_metrics.accuracy_score(y, p))
    assert measured["balanced_accuracy"] == pytest.approx(sk_metrics.balanced_accuracy_score(y, p))
    assert measured["precision_pneumonia"] == pytest.approx(sk_metrics.precision_score(y, p))
    assert measured["recall_pneumonia"] == pytest.approx(sk_metrics.recall_score(y, p))
    assert measured["f1_pneumonia"] == pytest.approx(sk_metrics.f1_score(y, p))
    assert measured["confusion_matrix"] == sk_metrics.confusion_matrix(y, p, labels=[0, 1]).tolist()
    assert measured["false_negatives"] == 1
    assert measured["false_positives"] == 2


def test_majority_uses_training_labels_and_ties_are_deterministic():
    train_labels = np.array([1, 1, 1, 0])
    unrelated_test_labels = np.array([0, 0, 0, 1])
    prediction = pg.majority_class(train_labels)
    assert prediction == 1
    # A different majority on test must have no role in the baseline fit.
    measured = pg.classification_metrics(unrelated_test_labels,
                                         np.full(len(unrelated_test_labels), prediction))
    assert measured["accuracy"] == .25
    assert measured["false_negatives"] == 0
    assert measured["false_positives"] == 3
    assert pg.majority_class(np.array([0, 1])) == 0
    with pytest.raises(ValueError):
        pg.majority_class(np.array([]))


def test_policy_exact_boundaries_and_forced_choice_equivalence():
    policy = pg.Policy(.2, .8, .05, 0, None)
    np.testing.assert_array_equal(
        pg.apply_policy(np.array([0., .199, .2, .5, .799, .8, 1.]), policy),
        [0, 0, -1, -1, -1, 1, 1],
    )
    scores = np.array([0., .1, .499999, .5, .500001, 1.])
    np.testing.assert_array_equal(pg.apply_policy(scores, replace(policy, low=.5, high=.5)),
                                  (scores >= .5).astype(int))


def test_zero_coverage_is_undefined_error_not_perfect_accuracy():
    batch = score_batch("test", [0, 1], [.1, .9])
    measured = pg.selective_metrics(batch, pg.Policy(0, 1, .05, 0, None, abstain_all=True))
    assert measured["coverage"] == 0
    assert measured["n_review"] == 2
    assert measured["automatic_error_rate"] is None
    assert measured["automatic_only"]["accuracy"] is None
    assert measured["automatic_only"]["recall_pneumonia"] is None
    assert measured["reference_by_decision_matrix"] == [[0, 0, 1], [0, 0, 1]]


def test_undefined_denominators_and_absent_reference_class_are_explicit():
    measured = pg.classification_metrics(np.array([0, 0]), np.array([0, 0]))
    assert measured["accuracy"] == 1
    assert measured["precision_pneumonia"] is None
    assert measured["recall_pneumonia"] is None
    assert measured["balanced_accuracy"] is None
    assert measured["f1_pneumonia"] is None


def test_selective_denominators_and_review_classes_do_not_count_as_fn():
    batch = score_batch("test", [0, 0, 1, 1, 1, 0], [.1, .9, .4, .6, .95, .45])
    measured = pg.selective_metrics(batch, pg.Policy(.3, .8, .05, 0, None))
    assert measured["coverage"] == .5
    assert measured["n_automatic"] == 3
    assert measured["automatic_error_rate"] == pytest.approx(1 / 3)
    assert measured["automatic_errors_per_all_cases"] == pytest.approx(1 / 6)
    assert measured["abstained_normal"] == 1
    assert measured["abstained_pneumonia"] == 2
    assert measured["automatic_only"]["false_negatives"] == 0
    assert measured["automatic_only"]["false_positives"] == 1
    assert measured["reference_by_decision_matrix"] == [[1, 1, 1], [0, 1, 2]]


@pytest.mark.parametrize("split,id_split", [("test", "test"), ("train", "train"),
                                                ("val", "test"), ("val", "train")])
def test_policy_selection_rejects_nonvalidation_sources(split, id_split):
    with pytest.raises(ValueError, match="validation"):
        pg.select_policy(score_batch(split, [0, 1], [.1, .9], id_split=id_split))


def test_grid_selection_maximizes_coverage_with_deterministic_ties():
    # To make no mistakes, exclude the two low-scoring positives and .6 negative.
    # The largest error-free accepted set has 5/8 images. Several thresholds yield
    # that set; lexicographic tie breaking selects low=.25, high=.65.
    batch = score_batch("val", [0, 0, 1, 1, 0, 1, 1, 1],
                        [.1, .2, .3, .4, .6, .7, .8, .9])
    first, grid = pg.select_policy(batch, target_error=0)
    second, _ = pg.select_policy(batch, target_error=0)
    assert first == second
    assert first.low == .25
    assert first.high == .65
    assert first.validation_coverage == 5 / 8
    assert first.validation_error == 0
    assert len(grid) == 121


def test_policy_falls_back_to_all_review_if_no_nonempty_feasible_set():
    # Both scores are confidently wrong for their reference class.
    policy, _ = pg.select_policy(score_batch("val", [1, 0], [.1, .9]), target_error=0)
    assert policy.abstain_all
    assert policy.validation_error is None
    assert policy.validation_coverage == 0
    np.testing.assert_array_equal(pg.apply_policy(np.array([0, .5, 1]), policy), [-1, -1, -1])


@pytest.mark.parametrize("scores", [[np.nan], [np.inf], [-.01], [1.01]])
def test_invalid_model_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        score_batch("val", [0], scores)


def test_image_scaling_model_shape_and_seed_are_reproducible():
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[1] = 255
    tensor = pg.image_tensor(images)
    assert tensor.shape == (2, 1, 28, 28)
    assert tensor.dtype == torch.float32
    assert torch.all(tensor[0] == 0)
    assert torch.all(tensor[1] == 1)
    pg.set_seed(2026)
    first = pg.SmallCNN()(tensor).detach()
    pg.set_seed(2026)
    second = pg.SmallCNN()(tensor).detach()
    assert first.shape == (2,)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.fixture
def tiny_trained_run(monkeypatch, tmp_path):
    """One tiny synthetic epoch, solely to inspect pipeline access order."""
    calls = []
    output = tmp_path / "results"
    data = tmp_path / "data"
    data.mkdir()
    (data / "pneumoniamnist.npz").write_bytes(b"synthetic test fixture, not public dataset")

    def fake_load_split(name, root, limit=None):
        calls.append(name)
        if name == "test":
            assert (output / "policy.json").is_file()
            assert (output / "best_model.pt").is_file()
            assert (output / "training_manifest.json").is_file()
        offset = {"train": 0, "val": 10, "test": 20}[name]
        images = np.stack([np.full((28, 28), offset + i, dtype=np.uint8) for i in range(4)])
        return pg.ImageSplit(name, images, np.array([0, 1, 1, 1]),
                              np.array([f"{name}:{i}" for i in range(4)]))

    monkeypatch.setattr(pg, "load_split", fake_load_split)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = pg.Config(max_epochs=1, patience=1, batch_size=4, data_dir=str(data), output_dir=str(output))
    experiment = pg.train_and_freeze(config)
    return experiment, calls, output


def test_training_never_opens_test_and_final_evaluation_uses_frozen_policy(tiny_trained_run):
    experiment, calls, output = tiny_trained_run
    assert calls == ["train", "val"]
    assert experiment.test is None
    frozen_policy_hash = pg.file_sha256(output / "policy.json")
    pg.evaluate_test(experiment)
    assert calls == ["train", "val", "test"]
    assert pg.file_sha256(output / "policy.json") == frozen_policy_hash
    assert experiment.metrics["baseline"]["false_positives"] == 1
    assert experiment.metrics["baseline"]["false_negatives"] == 0
    # An already evaluated run returns the recorded result, without test retuning.
    pg.evaluate_test(experiment)
    assert calls == ["train", "val", "test"]


def test_changed_frozen_policy_is_rejected_before_test_access(tiny_trained_run):
    experiment, calls, output = tiny_trained_run
    with (output / "policy.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="policy changed"):
        pg.evaluate_test(experiment)
    assert calls == ["train", "val"]


def test_changed_checkpoint_is_rejected_before_test_access(tiny_trained_run):
    experiment, calls, output = tiny_trained_run
    with (output / "best_model.pt").open("ab") as stream:
        stream.write(b"tampered checkpoint fixture")
    with pytest.raises(ValueError, match="checkpoint changed"):
        pg.evaluate_test(experiment)
    assert calls == ["train", "val"]


def test_demo_performs_fresh_inference_and_changes_when_model_weights_change():
    class ConstantLogit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(-3.))
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return self.bias.expand(images.shape[0])

    model = ConstantLogit()
    split = pg.ImageSplit("test", np.zeros((1, 28, 28), dtype=np.uint8),
                           np.array([1]), np.array(["test:0"]))
    experiment = SimpleNamespace(test=split, model=model, device=torch.device("cpu"),
                                 policy=pg.Policy(.2, .8, .05, 0, None))
    first = pg.infer_test_image(experiment, 0)
    with torch.no_grad():
        model.bias.fill_(3.)
    second = pg.infer_test_image(experiment, 0)
    assert model.calls == 2
    assert first["forced_prediction"] == "Normal"
    assert second["forced_prediction"] == "Pneumonia"
    assert first["pneumonia_score"] < second["pneumonia_score"]
    assert first["automatic_matches_reference"] is False
    assert second["automatic_matches_reference"] is True
    with pytest.raises(ValueError, match="index"):
        pg.infer_test_image(experiment, 1)
