# Model Card: AI Chest X-Ray Risk Awareness System

## Model Details
- **Developed by:** [Your team/name], Data Odyssey 2026
- **Model type:** CNN (DenseNet121, transfer learning from ImageNet)
- **Task:** Binary classification (Normal vs. Pneumonia-pattern), converted to a
  Low/Medium/High risk score
- **Explainability:** Grad-CAM heatmaps highlighting regions influencing the prediction

## Intended Use
- **Primary intended use:** Educational/awareness prototype demonstrating how AI +
  explainability could support earlier consultation for possible lung abnormalities.
- **Primary intended users:** Hackathon judges, students exploring health-AI literacy,
  and as a proof-of-concept for further clinical-grade development.

## Out-of-Scope Use
- **Not for clinical diagnosis.** Not validated against clinical outcomes, not reviewed
  by a radiologist, not a certified medical device.
- **Not for use on any imaging modality other than frontal chest X-rays.**
- **Not intended to detect lung cancer** in its current trained form.
- **Not calibrated for any specific age group beyond the training data's population**
  (state your dataset's population here).

## Training Data
- **Dataset:** Kaggle "Chest X-Ray Images (Pneumonia)" (Paul Mooney), ~5,863 images,
  pediatric patients, Guangzhou Women and Children's Medical Center.
- **Classes:** NORMAL, PNEUMONIA (bacterial + viral combined)
- **Known imbalance:** ~3:1 Pneumonia:Normal in training split; addressed via class weighting.

## Evaluation Results
- Accuracy: 90%
- Recall (Pneumonia): 91%
- Precision (Pneumonia): 93%
- ROC-AUC: 0.944
- Operating threshold: 0.5 default (0.3 available for higher-recall screening mode)

## Explainability & Known Limitations
- Grad-CAM outputs were manually reviewed on a sample of test images; heatmaps
  generally concentrated on lung fields as expected. [Add any specific observations
  from your gradcam_sanity_check.png here.]
- Low/Medium/High thresholds (30%/70%) are heuristic, not clinically calibrated.
- The "region" shown (e.g., "lower right lung field") is a coarse quadrant
  approximation from the heatmap, not precise anatomical localization.
- Small dataset size may limit generalization to X-rays from other hospitals/scanners.
- The model has no mechanism to detect whether an uploaded image is a chest X-ray
  at all — a non-X-ray image will still receive a probability score (known limitation).

## Ethical Considerations
- Designed to keep a human (doctor) in the loop — flags risk for awareness, never
  issues a final verdict.
- UI language avoids diagnostic claims ("you have pneumonia") in favor of
  risk-awareness framing ("patterns consistent with possible pneumonia").
- No patient-identifying data collected or stored; images processed transiently.

## Future Work
- Clinical validation with radiologist-reviewed ground truth
- Expansion to lung cancer risk detection
- Multi-institution data for improved generalization
- Formal threshold calibration against outcome data
