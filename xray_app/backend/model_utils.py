import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from datetime import datetime

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


def validate_image(img_path, min_size=50, blur_threshold=15.0):
    try:
        img = PILImage.open(img_path)
        img.verify()
    except Exception:
        return False, "The uploaded file could not be read as a valid image."

    img = PILImage.open(img_path)

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


def get_confidence_level(probability):
    distance_from_uncertain = abs(probability - 0.5)

    if distance_from_uncertain >= 0.35:
        return "High", "The model is confident in this assessment."
    elif distance_from_uncertain >= 0.15:
        return "Moderate", "The model has moderate confidence. Consider this alongside other information."
    else:
        return "Low", "This case falls in an uncertain range — treat this result with extra caution."


def get_heatmap_region(heatmap):
    h, w = heatmap.shape
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    vertical_zone = "upper zone" if y < h / 2 else "lower zone"
    side = "right" if x < w / 2 else "left"

    return f"{side} {vertical_zone}"


def generate_explanation(risk_level, probability, region):
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


def _wrap_text(text, max_chars):
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def generate_pdf_report(result, overlay_path, report_path):
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Chest X-Ray Screening Report")

    c.setFont("Helvetica", 10)
    c.drawString(1 * inch, height - 1.3 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, height - 1.8 * inch, f"Risk Level: {result['risk_level']}")
    c.setFont("Helvetica", 11)
    c.drawString(1 * inch, height - 2.1 * inch, f"Probability (pneumonia-associated patterns): {result['probability']:.1%}")
    c.drawString(1 * inch, height - 2.4 * inch, f"Model Confidence: {result['confidence_level']}")

    # Explanation text block — give it enough vertical room before the image starts
    c.setFont("Helvetica", 10)
    text_obj = c.beginText(1 * inch, height - 2.9 * inch)
    text_obj.setFont("Helvetica", 10)
    wrapped_lines = _wrap_text(result["explanation"], 90)
    for line in wrapped_lines:
        text_obj.textLine(line)
    c.drawText(text_obj)

    # Calculate where the text block ended, add spacing, THEN place the image below it
    line_height = 12  # approx points per line at font size 10
    text_block_height = len(wrapped_lines) * line_height
    image_top = height - 2.9 * inch - text_block_height - 0.4 * inch  # extra gap after text

    try:
        image_size = 3.2 * inch
        c.drawImage(
            overlay_path,
            1 * inch,
            image_top - image_size,
            width=image_size,
            height=image_size
        )
    except Exception:
        pass

    c.setFont("Helvetica-Oblique", 8)
    disclaimer = (
        "This is an AI screening aid, not a medical diagnosis. "
        "Please consult a qualified healthcare professional for clinical evaluation."
    )
    c.drawString(1 * inch, 1 * inch, disclaimer)

    c.save()
    return report_path


def predict_and_explain(img_path, output_path):
    img_array = preprocess_single_image(img_path)
    heatmap, pred_prob = make_gradcam_heatmap(img_array)
    overlay_img = overlay_gradcam(img_path, heatmap)

    PILImage.fromarray(overlay_img).save(output_path)

    risk_level = get_risk_level(pred_prob)
    region = get_heatmap_region(heatmap)
    explanation = generate_explanation(risk_level, pred_prob, region)
    confidence_level, confidence_note = get_confidence_level(pred_prob)

    return {
        "probability": float(pred_prob),
        "risk_level": risk_level,
        "explanation": explanation,
        "confidence_level": confidence_level,
        "confidence_note": confidence_note,
        "overlay_path": output_path
    }