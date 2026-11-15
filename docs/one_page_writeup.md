# PneumoGuard AI

**Veritas AI International High School Hackathon 2026 · AI + Health**

PneumoGuard AI investigates whether a small image classifier can make fewer incorrect automatic decisions by referring uncertain public benchmark images for manual review.

**Research question.** Can an uncertainty-aware image classifier reduce incorrect automatic decisions compared with a standard forced-choice classifier, and what is the trade-off in manual review workload?

**Problem and data.** A hypothetical healthcare worker may need to prioritize images for review. This educational experiment studies that idea using official 28 × 28 grayscale PneumoniaMNIST images and predefined training, validation, and test splits. Original labels remain 0 = Normal and 1 = Pneumonia. Class counts are generated during execution. Pediatric origin, low resolution, class imbalance, and a restricted source population limit generalization. Benchmark annotations are not independent clinical confirmation. Private scans are never accepted.

**Method.** A baseline always predicts the most common training label. A compact PyTorch CNN uses two convolution layers with 16 and 32 feature channels and a classification head. Only training images update weights. Validation performance selects the checkpoint, with at most 20 epochs and early stopping after four epochs without improvement. The random seed is 2026; no GPU or paid API is required. The standard CNN uses a fixed 0.5 threshold. Its scores are not calibrated medical probabilities.

**Original contribution.** A validation-only grid search selects two acceptance thresholds. Scores below the lower threshold favor Normal; scores at or above the upper threshold favor Pneumonia; others trigger review. The rule maximizes validation coverage subject to an empirical accepted-image error rate at most 5%, with deterministic tie-breaking and an all-review fallback. This research setting is not a safety guarantee. The policy is frozen before final test evaluation. The contribution connects abstention, measured errors, and manual-review workload in a reproducible experiment and live benchmark demonstration.

**Results and interpretation.** **Not yet evaluated** until a full run generates `results/report.md`; use that report for all submission numbers. It compares the baseline, forced-choice CNN, and abstaining CNN with confusion matrices, false negatives, false positives, precision, recall, F1, accuracy, balanced accuracy, and actual failure examples. Selective metrics describe accepted images only; complete-test coverage and abstained class counts reveal the remaining workload. These denominators must be reported together. Any lack of improvement must be reported honestly. No clinical validation or patient-safety claim follows from this benchmark.

**Reproduction.** Open `notebooks/PneumoGuard_AI.ipynb` in Google Colab and run all cells. It installs dependencies, loads data, trains, freezes thresholds, evaluates, exports measured artifacts, and displays actual inference on selected public test images. Keep smoke-test results separate.

**Sources and credits.** [MedMNIST](https://github.com/MedMNIST/MedMNIST), [data distribution](https://zenodo.org/records/10519652), [MedMNIST v2 paper](https://doi.org/10.1038/s41597-022-01721-8), [Kermany et al.](https://doi.org/10.1016/j.cell.2018.02.010), and [source data, CC BY 4.0](https://doi.org/10.17632/rscbjbr9sj.2). Libraries and environment are credited in the repository. ChatGPT and OpenAI Codex assisted with implementation and documentation; the student must review and understand the work and follow the competition’s AI-disclosure rules.

This project is a prototype and is not medical advice. Predictions are for research and educational purposes only and must not be used for clinical decisions.
