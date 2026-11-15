# Final submission checklist

Checked evidence items refer to the archived full run and its extracted files. They do not claim that training or a live Colab demonstration was independently repeated during this update. No retraining is required to prepare this submission.

## Saved evidence

- [x] Preserve the original full-run ZIP and extract its measured artifacts into [results/](../results/), including checkpoint, configuration, predictions, environment, report and three figures.
- [x] Record the full mode (`smoke=false`), seed 2026, 28 × 28 inputs and original labels: 0 = Normal, 1 = Pneumonia.
- [x] Record split counts and classes: training 4,708 (1,214 Normal / 3,494 Pneumonia), validation 524 (135 / 389), test 624 (234 / 390).
- [x] Preserve the training manifest: best epoch 20, training/validation selection only, `test_used_for_selection=false`, checkpoint and policy hashes. This is saved run evidence, not independent observation of training.
- [x] Preserve validation predictions and the policy grid, the 5% empirical error target, deterministic selection and all-review fallback.
- [x] Report eight distinct shared image hashes between training and validation; none shared between train–test or validation–test. Patient-level independence remains unverifiable.
- [x] Report the training-majority baseline: Pneumonia, 62.50% test accuracy and 50.00% balanced accuracy.
- [x] Report CNN accuracy 82.85% on 624 test images, balanced accuracy 77.56%, precision 79.06%, recall 98.72%, F1 87.80%, five FN and 102 FP. Precision/recall/F1 concern Pneumonia.
- [x] Report `low=high=0.5`, 624/624 automatic decisions (100% coverage), 0/624 reviews, and 107/624 errors (17.15%) for both CNN policies. Abstention gave no improvement.
- [x] Include five archived real mistakes and explain `test:1` without inventing a medical cause.
- [x] Keep full-run evidence distinct from smoke or synthetic development fixtures.

## Student review and presentation

- [ ] Read the README, write-up and saved report; understand and explain every model and policy choice, including the lack of improvement and validation overlap.
- [ ] Review the concrete [demo narration](demo_script.md), rehearse and, if desired, record an optional video of up to two minutes. Saved-result presentation does not require a fresh run or live inference.
- [ ] If an executed notebook was saved from the completed Colab session, retain that existing copy as additional evidence; do not rerun training just to obtain it.
- [x] Include “This project is a prototype and is not medical advice.” and the research-only restriction.
- [x] Explain pediatric origin, low resolution, limited source population, uncalibrated scores, annotation-based labels and lack of clinical validation.
- [x] Retain dataset, source, library and AI-tool credits, including ChatGPT and OpenAI Codex, in [sources.md](sources.md).
- [ ] Review data licenses and retain attribution with shared example figures or artifacts.

## Official requirements and manual submission

- [x] Recheck the official [submission page](https://highschoolhackathon.org/briefs) and [FAQ](https://highschoolhackathon.org/faq) on November 15, 2026. Required: viewable code/Colab link, one write-up, track, team details and linked credits. Health projects must carry the medical disclaimer.
- [ ] Read the full [official project-brief PDF](https://drive.google.com/file/d/15tYD48XRmCgWbk1_lLDtc8lebi6qm-4-/view?usp=sharing). It still returned only loading/sign-in text during this review; complete Health-track compliance is not verified.
- [ ] Confirm every participant was enrolled in high school on November 14, the team has 1–3 contributing members, and registration was completed. The published registration deadline was November 13.
- [ ] Confirm project code was written within November 14, 00:00 ET–November 15, 23:59 ET and did not extend a pre-hackathon project. Check the actual creation history; do not invent provenance.
- [ ] Review the prepared pull request in the existing repository and merge it when ready. Continue using this repository; no new repository is needed.
- [ ] Open the repository, notebook, results and write-up links in a private window and verify they are viewable without signing in. Opening the notebook does not require running all cells.
- [ ] Enter accurate full name, email and school for every teammate, choose Health and disclose all tools and sources.
- [ ] Submit manually through the official form by **November 15, 2026, 11:59 PM ET** (**November 16, 09:59 in Asia/Qyzylorda**). Verify the form's current availability yourself; late submissions are not accepted according to the FAQ.

The assistant does not submit the competition entry. No award, endorsement, medical certification or clinical effectiveness is claimed.
