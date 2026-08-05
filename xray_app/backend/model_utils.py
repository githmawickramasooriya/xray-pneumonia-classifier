import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_final.keras")

model = load_model(MODEL_PATH)

base_model_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model_layer = layer
        break

last_conv_layer_name = base_model_layer.layers[-1].name

grad_model = tf.keras.models.Model(
    inputs=base_model_layer.input,
    outputs=[
        base_model_layer.get_layer(last_conv_layer_name).output,
        base_model_layer.output
    ]
)

head_input = tf.keras.Input(shape=base_model_layer.output.shape[1:])
x = head_input
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        continue
    if "input" in layer.name.lower():
        continue
    x = layer(x)
head_model = tf.keras.Model(head_input, x)

print("Grad-CAM models built and ready.")


# --- Input validation (runs before the model ever sees the image) ---
def validate_image(img_path, min_size=50, blur_threshold=15.0):
    """
    Basic sanity checks on an uploaded image before running inference.
    Returns (is_valid: bool, reason: str or None)
    """
    try:
        img = PILImage.open(img_path)
        img.verify()
    except Exception:
        return False, "The uploaded file could not be read as a valid image."

    img = PILImage.open(img_path)  # reopen — verify() closes the file handle

    width, height = img.size
    if width < min_size or height < min_size:
        return False, f"Image resolution too low ({width}x{height}). Please upload a clearer image."

    img_gray = np.array(img.convert("L"))
    std_dev = img_gray.std()
    if std_dev < 5:
        return False, "This image appears blank or has very little detail — please check the file."

    laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False, "This image appears too blurry for reliable analysis. Please upload a sharper image."

    return True, None


def preprocess_single_image(img_path, target_size=IMG_SIZE):
    img = image.load_img(img_path, target_size=target_size, color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


def make_gradcam_heatmap(img_array):
    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_array)
        tape.watch(conv_output)
        preds = head_model(base_output)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), preds.numpy()[0][0]


def overlay_gradcam(img_path, heatmap, alpha=0.4, colormap="jet"):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (img.shape[0], img.shape[1]))
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()

    cmap = plt.colormaps[colormap]
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    superimposed = colored_heatmap * alpha + img * (1 - alpha)
    return np.uint8(superimposed)


def get_risk_level(probability, low_threshold=0.30, high_threshold=0.70):
    if probability < low_threshold:
        return "Low"
    elif probability < high_threshold:
        return "Medium"
    else:
        return "High"


def get_heatmap_region(heatmap):
    """
    Coarse quadrant-based approximation of where the heatmap is most active.
    Not a precise anatomical localization — see MODEL_CARD.md for this limitation.
    """
    h, w = heatmap.shape
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    vertical = "upper" if y < h / 2 else "lower"
    horizontal = "right" if x < w / 2 else "left"  # X-ray convention: image-left = patient's right

    return f"{vertical} {horizontal} lung field"


def generate_explanation(risk_level, probability, region):
    """
    Plain-language summary of the MODEL'S OUTPUT — describes what the model
    flagged, framed as awareness/screening language, never as a diagnosis.
    """
    disclaimer_tail = (
        " This is an AI screening aid, not a diagnosis. "
        "Please consult a qualified healthcare professional for clinical evaluation."
    )

    if risk_level == "Low":
        return (
            "The model did not flag strong pneumonia-associated patterns in this image; "
            "the lung fields appear largely clear to the model." + disclaimer_tail
        )
    elif risk_level == "Medium":
        return (
            f"The model flagged some irregularity, most notably around the {region}, "
            f"though its confidence is moderate ({probability:.0%}). "
            "A specialist review may be warranted." + disclaimer_tail
        )
    else:
        return (
            f"The model flagged patterns often associated with pneumonia, "
            f"particularly around the {region} ({probability:.0%} model confidence)." + disclaimer_tail
        )


def predict_and_explain(img_path, output_path):
    img_array = preprocess_single_image(img_path)
    heatmap, pred_prob = make_gradcam_heatmap(img_array)
    overlay_img = overlay_gradcam(img_path, heatmap)

    PILImage.fromarray(overlay_img).save(output_path)

    risk_level = get_risk_level(pred_prob)
    region = get_heatmap_region(heatmap)
    explanation = generate_explanation(risk_level, pred_prob, region)

    return {
        "probability": float(pred_prob),
        "risk_level": risk_level,
        "explanation": explanation,
        "overlay_path": output_path
    }