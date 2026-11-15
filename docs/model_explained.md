# Explaining the experiment

## What the CNN learns

Each input is one public grayscale image with 28 × 28 pixels. Two convolution layers, with 16 and 32 feature channels, learn visual patterns from training images. A small classification head converts the learned features to one score. The model begins with random weights; it does not use a hidden diagnostic service or hand-written image answers.

A binary classification loss compares the training output with the original dataset label. Updating the weights reduces that loss. Validation images help select the saved checkpoint. They do not update the weights. The test split is reserved for the final evaluation.

The model score lies between 0 and 1. Larger scores favor the Pneumonia benchmark label; smaller scores favor Normal. A score is not a calibrated medical probability, and “Normal” here is a benchmark category, not reassurance about a real person's health.

## Three systems to compare

| System | How it decides | Images it answers |
|---|---|---|
| Majority baseline | Always predicts the most common label in training | All test images |
| Standard CNN | Uses the trained score and a fixed 0.5 threshold | All test images |
| CNN with abstention | Uses validation-selected acceptance thresholds | Only accepted test images; the others require review |

The thresholds are selected on validation scores and labels and saved before the test predictions are evaluated. The fixed search tries lower thresholds 0, 0.05, …, 0.5 and upper thresholds 0.5, 0.55, …, 1. A score below the lower threshold is accepted as Normal; a score at or above the upper threshold is accepted as Pneumonia; other scores are referred for review. The rule selects the greatest validation coverage among candidates with an empirical accepted-image error rate at most 5%. Ties prefer lower error, then ascending lower and upper thresholds. If no candidate qualifies, it abstains on all images.

The 5% setting is a research design choice, not a clinical safety limit or a promise about future errors. The chosen values appear in `results/policy.json` and the measured report. The same test results must not be used to try thresholds until the numbers look better.

## False negatives and false positives

A **false negative** is an image with reference label Pneumonia that the system automatically classifies as Normal. A **false positive** is an image with reference label Normal that it automatically classifies as Pneumonia. Both are disagreements with dataset annotations; neither establishes a real patient diagnosis.

For the abstaining system, an image referred for review is neither an automatic false negative nor an automatic false positive. It remains part of the total workload. Therefore, always report abstained class counts alongside the accepted confusion matrix.

A confusion matrix uses reference labels as rows and predicted labels as columns:

| Reference / prediction | Normal | Pneumonia |
|---|---:|---:|
| Normal | True negatives | False positives |
| Pneumonia | False negatives | True positives |

## Reading the results

- **Accuracy:** fraction of predictions that match the reference labels. A common class can make this look strong even when the other class is missed.
- **Balanced accuracy:** average of recall for each class, so the two classes receive equal weight.
- **Pneumonia recall:** fraction of reference-positive images predicted positive in the evaluated subset.
- **Pneumonia precision:** fraction of positive predictions that have the positive reference label.
- **F1:** combines precision and recall. It does not describe how many images were deferred.
- **Coverage:** accepted images divided by all test images.
- **Manual-review fraction:** abstained images divided by all test images.
- **Automatic error rate:** incorrect automatic decisions divided by accepted images.

For the baseline and standard CNN, the evaluated subset is the full test set. For selective classification metrics, it contains accepted images only. A selective recall value does not show what happened to reference-positive images that were abstained. The report therefore also includes coverage, review workload, and the reference-class distribution among abstentions. If no images or no relevant class are accepted, a metric may be undefined; consult the report's metric conventions.

Lowering the error rate among accepted images by refusing more images can increase manual-review workload substantially. It does not guarantee safety or prove that the selected policy transfers to another population. The validation split is small, the images are low resolution, and there is no prospective clinical evaluation.

This project is a prototype and is not medical advice. Predictions are for research and educational purposes only and must not be used for clinical decisions.
