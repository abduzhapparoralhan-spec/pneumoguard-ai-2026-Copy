# Final submission checklist

This checklist is completed by the student. Its unchecked items are not claims that verification has already happened.

## Reproduction and evidence

- [ ] Open `notebooks/PneumoGuard_AI.ipynb` in a fresh Google Colab session and run all cells in full mode.
- [ ] Confirm the data are official PneumoniaMNIST at 28 × 28 resolution and the original labels remain 0 = Normal, 1 = Pneumonia.
- [ ] Confirm training, validation, and test counts and class imbalance are recorded.
- [ ] Confirm the model trained on training images only and checkpoint selection used validation data only.
- [ ] Confirm the abstention thresholds were selected using validation data and frozen before final test evaluation; explain the fixed grid, 5% empirical validation-error target, tie-breaking rule, and all-review fallback.
- [ ] Save measured results, the trained checkpoint, configuration, dependency versions, and the executed notebook.
- [ ] Confirm the baseline is the majority class measured in the training split.
- [ ] Confirm accuracy, balanced accuracy, Pneumonia precision, recall, F1, confusion matrices, false negatives, and false positives are visible.
- [ ] Report selective metrics together with their accepted-image denominator, complete-test coverage, manual-review workload, and abstained class counts.
- [ ] Discuss at least five real mistakes or ambiguous examples when available; explain one observed failure without inventing its medical cause.
- [ ] Confirm every claim in the write-up and narration matches the generated full-run report. Mark unexecuted work “Not yet evaluated.”
- [ ] State whether abstention actually improved the chosen error metric; do not imply that a change is guaranteed to repeat.
- [ ] Keep smoke-test output separate from the final experiment.

## Demonstration and communication

- [ ] Demonstrate a public test image producing an actual model inference, a category, an accept/abstain status, and the reference label.
- [ ] Replace all narration placeholders with measured numbers and rehearse a two-minute explanation.
- [ ] Show the medical disclaimer prominently and include the exact sentence: “This project is a prototype and is not medical advice.”
- [ ] Explain pediatric origin, low image resolution, limited source population, uncalibrated scores, and lack of clinical validation.
- [ ] Explain that benchmark labels are annotations, not independent clinical confirmation.
- [ ] Include dataset, underlying source, library, and AI-tool credits, including ChatGPT and OpenAI Codex.
- [ ] Review relevant data licenses before sharing images, results, or model artifacts.

## Manual publication and submission

- [ ] Check the official [competition requirements](https://highschoolhackathon.org/briefs), submission deadline, AI-assistance policy, and eligibility yourself. The briefs page was accessible during development, but its linked detailed document required sign-in/loading; implementation therefore follows the requirements in the supplied project brief.
- [ ] Confirm the project’s creation history satisfies the competition’s original-work period; do not invent evidence of provenance.
- [ ] Create or update the public repository manually; exclude dataset downloads, caches, credentials, and private information.
- [ ] Open the public repository while signed out and verify that README, notebook, measured results, and write-up are readable.
- [ ] Test the Colab notebook from the public repository link.
- [ ] Submit the repository/notebook link, concise write-up, and requested presentation materials through the official submission process.

The assistant does not automatically publish or submit this project. No award, endorsement, medical certification, or clinical effectiveness is claimed.
