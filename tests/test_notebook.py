"""Notebook portability and scientifically honest presentation regression checks."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import nbformat
import numpy as np
import pandas as pd

from src import pneumoguard as pg


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "PneumoGuard_AI.ipynb"


def read_notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def test_notebook_has_valid_schema_and_every_python_cell_compiles():
    notebook = read_notebook()
    nbformat.validate(notebook)
    assert notebook.metadata.kernelspec.language == "python"
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"PneumoGuard_AI.ipynb:cell-{index}", "exec")


def test_embedded_implementation_matches_repository_source_exactly():
    cells = [cell for cell in read_notebook().cells
             if "core-implementation" in cell.metadata.get("tags", [])]
    assert len(cells) == 1
    module_source = (ROOT / "src" / "pneumoguard.py").read_text(encoding="utf-8")
    library_source = module_source.split('\nif __name__ == "__main__":', 1)[0]
    assert cells[0].source.strip() == library_source.strip()


def test_notebook_installation_precedes_training_evaluation_and_live_demo():
    notebook = read_notebook()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "dependencies" in code_cells[0].metadata.get("tags", [])
    assert "pip" in code_cells[0].source
    assert "medmnist==3.0.2" in code_cells[0].source
    tagged = {tag: index for index, cell in enumerate(notebook.cells)
              for tag in cell.metadata.get("tags", [])}
    assert tagged["dependencies"] < tagged["core-implementation"] < tagged["configuration"]
    assert tagged["configuration"] < tagged["training-and-policy"] < tagged["final-evaluation"]
    assert tagged["final-evaluation"] < tagged["interactive-demo"]
    assert "train_and_freeze(config)" in notebook.cells[tagged["training-and-policy"]].source
    assert "evaluate_test(experiment)" in notebook.cells[tagged["final-evaluation"]].source
    assert "show_demo(experiment)" in notebook.cells[tagged["interactive-demo"]].source


def test_notebook_source_has_no_codex_machine_specific_paths():
    source = "\n".join(cell.source for cell in read_notebook().cells)
    for forbidden_path in ["/workspace", "/codex/", "/home/agent", "/tmp/"]:
        assert forbidden_path not in source
    # A fresh Colab session can execute the embedded implementation on its own.
    assert "from src" not in source
    assert "import src" not in source


def test_plotting_no_errors_and_zero_coverage_does_not_invent_success_or_failure(monkeypatch, tmp_path):
    """Synthetic presentation fixture, never a measured benchmark result."""
    from matplotlib.axes import Axes

    labels = np.array([0, 0, 1, 1, 1])
    scores = np.array([.1, .2, .8, .9, .95])
    batch = pg.ScoreBatch("test", labels, scores, np.array([f"test:{i}" for i in range(5)]))
    policy = pg.Policy(0, 1, .05, 0, None, abstain_all=True)
    experiment = SimpleNamespace(
        config=pg.Config(output_dir=str(tmp_path)),
        policy=policy,
        test=pg.ImageSplit("test", np.zeros((5, 28, 28), dtype=np.uint8), labels, batch.ids),
        test_batch=batch,
        metrics={"baseline": pg.classification_metrics(labels, np.ones(5, dtype=int)),
                 "cnn_forced": pg.classification_metrics(labels, (scores >= .5).astype(int)),
                 "cnn_selective": pg.selective_metrics(batch, policy)},
        history=pd.DataFrame([{"epoch": 1, "train_loss": .5, "val_loss": .4}]),
        best_epoch=1,
        validation_grid=pd.DataFrame([{"coverage": 0., "automatic_error_rate": None}]),
    )
    scatter_labels, texts = [], []
    original_scatter, original_text = Axes.scatter, Axes.text

    def record_scatter(self, *args, **kwargs):
        scatter_labels.append(kwargs.get("label"))
        return original_scatter(self, *args, **kwargs)

    def record_text(self, x, y, text, *args, **kwargs):
        texts.append(str(text))
        return original_text(self, x, y, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Axes, "text", record_text)
    examples = pg.create_figures(experiment)

    assert "Selected on validation" not in scatter_labels
    assert any("no automatic decisions; error undefined" in text for text in texts)
    assert len(examples) == 5
    assert set(examples["kind"]) == {"Closest to boundary (correct)"}
    assert set(examples["policy_decision"]) == {"Review"}
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## One observed failure" not in report
    assert "0 forced-choice mistakes were found" in report
    assert "no forced-choice errors found" in report
    for figure in ["confusion_matrices.png", "training_and_validation_tradeoff.png", "failure_examples.png"]:
        assert (tmp_path / "figures" / figure).stat().st_size > 0
