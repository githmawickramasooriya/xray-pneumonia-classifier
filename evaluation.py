import os
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the trained model ---
model = tf.keras.models.load_model("best_model_final.keras")

# --- 2. Recreate test generator ---
base_dir = r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\archive (1)\chest_xray"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False   # keep order so predictions line up with true labels
)

# --- 3. Get predictions ---
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

# --- 4. Classification report ---
print(classification_report(y_true, y_pred, target_names=["NORMAL", "PNEUMONIA"]))

# --- 5. Confusion matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["NORMAL", "PNEUMONIA"],
            yticklabels=["NORMAL", "PNEUMONIA"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# False negatives = missed pneumonia cases — most dangerous error type
fn = cm[1][0]
print(f"False Negatives (missed pneumonia cases): {fn}")

# --- 6. ROC-AUC ---
auc_score = roc_auc_score(y_true, y_pred_prob)
print(f"ROC-AUC: {auc_score:.3f}")

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --- 7. Try lowering the decision threshold to prioritize recall ---
for threshold in [0.5, 0.4, 0.3, 0.2]:
    y_pred_t = (y_pred_prob >= threshold).astype(int)
    report = classification_report(
        y_true, y_pred_t, target_names=["NORMAL", "PNEUMONIA"], output_dict=True
    )
    print(f"Threshold {threshold}: "
          f"Recall(PNEUMONIA)={report['PNEUMONIA']['recall']:.3f}, "
          f"Precision(PNEUMONIA)={report['PNEUMONIA']['precision']:.3f}")


# --- 8. Convert probability to Low/Medium/High risk ---
def get_risk_level(probability, low_threshold=0.30, high_threshold=0.70):
    """
    Converts model's raw sigmoid output (probability of PNEUMONIA)
    into a patient-facing risk category.

    Thresholds are a judgment call — document reasoning:
    - Below low_threshold: model is fairly confident it's NORMAL
    - Above high_threshold: model is fairly confident it's PNEUMONIA
    - In between: genuine uncertainty zone, flagged for clinical review
    """
    if probability < low_threshold:
        return "Low", probability
    elif probability < high_threshold:
        return "Medium", probability
    else:
        return "High", probability


# Example usage on a single test image
sample_prob = y_pred_prob[0]
risk_label, score = get_risk_level(sample_prob)
print(f"Probability: {score:.2f} → Risk Level: {risk_label}")

# Apply across the whole test set to sanity-check distribution of risk buckets
risk_levels = [get_risk_level(p)[0] for p in y_pred_prob]
print(collections.Counter(risk_levels))

# --- 9. Save final model for later use (e.g. Flask/FastAPI backend) ---
model.save("pneumonia_risk_model.keras")
model = tf.keras.models.load_model(r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\best_model_final.keras")
plt.show()  
# comment this out temporarily to skip pausing
model.save(r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\pneumonia_risk_model.keras")
