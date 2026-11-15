# Two-minute demonstration narration

Read this after a full notebook run. Replace bracketed prompts with measured values from `results/report.md`; never read placeholders as results. Choose an actual failure shown by the notebook. If training has not finished, say “Not yet evaluated” and demonstrate only what has actually run.

**0:00–0:20 — The question**

“PneumoGuard AI asks a research question: can an image classifier make fewer incorrect automatic decisions if it is allowed to ask for manual review? This project is a prototype and is not medical advice. We use public benchmark images only.”

**0:20–0:45 — Data and learning**

“These are 28 by 28 pixel images from PneumoniaMNIST, a pediatric chest X-ray benchmark. Our small convolutional neural network learns from the training split. Validation images select its checkpoint and uncertainty thresholds. We freeze those choices before evaluating the unseen test split. The reference labels are dataset annotations, not our own clinical confirmation.”

**0:45–1:10 — Live inference**

[Select a public test-image index and run the prediction.]

“This image is passed through our trained model now. Here is the model score, its forced category, and the uncertainty policy’s decision. The score is not a calibrated probability of disease. The reference label lets us check the output. An uncertain result means the policy declines to make an automatic category decision.”

**1:10–1:35 — Comparison**

“Our baseline always chooses the most common training class. Its test balanced accuracy is [measured value]. The standard CNN has [measured false negatives] false negatives and [measured false positives] false positives. With abstention, [coverage percent] of test images receive automatic decisions and [review percent] are referred for review. The error rate among accepted images is [measured rate], compared with [forced-choice error rate] when the CNN must answer every time. These rates have different denominators.”

**1:35–2:00 — Failure and meaning**

[Show a real incorrect prediction or ambiguous example.]

“Here the model [describe the displayed score, decision, and reference-label mismatch]. This demonstrates [the specific observed failure; do not invent a medical explanation]. Our result [did/did not] reduce automatic error rate at the cost of additional manual review. A small benchmark cannot establish clinical safety, performance on adults, or performance in another hospital. Our contribution is a reproducible way to measure this trade-off honestly.”

If all predictions are correct in a particular run, say that no test mistake was observed and show an actual abstained example instead. Do not invent a failure.
