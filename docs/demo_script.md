# Two-minute demonstration narration

Use the completed run in [results/report.md](../results/report.md). Show its saved figures and predictions; this script demonstrates archived results and does not claim that inference is running live. No retraining is needed. Rehearse before recording; a video is optional.

**0:00–0:20 — The question**

“PneumoGuard AI asks whether a classifier can reduce automatic errors by referring uncertain images for review. This project is a prototype and is not medical advice. We use public benchmark images only.”

**0:20–0:45 — Data and learning**

Show the data and method sections of the README.

“The experiment uses 28 by 28 pixel pediatric PneumoniaMNIST images. Training updates model weights; validation selects the checkpoint and thresholds before final test evaluation. The saved manifest records eight distinct shared image hashes between training and validation, with none shared with test. Patient independence cannot be verified.”

**0:45–1:10 — Measured comparison**

Show [the confusion matrices](../results/figures/confusion_matrices.png) and [comparison table](../results/comparison.csv).

“On 624 test images, the majority baseline has 62.50 percent accuracy and 50 percent balanced accuracy. Our CNN reaches 82.85 percent accuracy and 77.56 percent balanced accuracy, with five false negatives and 102 false positives against the benchmark labels.”

**1:10–1:35 — What abstention actually did**

Show [policy.json](../results/policy.json) and the workload section of the report.

“Validation selected both thresholds at 0.5. The policy therefore accepted all 624 test images: 100 percent coverage and zero manual reviews. Its error rate was 17.15 percent, exactly the same as forced choice. Abstention did not improve this run. The five-percent validation target did not guarantee test performance.”

**1:35–2:00 — Failure and meaning**

Show the first example in [the saved failure figure](../results/figures/failure_examples.png), also recorded in [failure_examples.csv](../results/failure_examples.csv).

“For test image one, the reference label is Normal, but the model predicts Pneumonia with score 0.9999. That score is not a calibrated disease probability. We cannot infer a medical cause from this tiny image. One seed on one pediatric benchmark cannot establish clinical safety or performance on adults or other hospitals. We report the failed abstention improvement honestly.”

Reference labels are dataset annotations, not independent clinical confirmation. Keep the [dataset attribution](sources.md) with any shared image figure and disclose ChatGPT and OpenAI Codex assistance in the presentation credits.
