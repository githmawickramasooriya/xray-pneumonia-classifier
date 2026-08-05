## ===== Sanity-Check Grad-CAM Across Multiple Samples =====
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage

base_dir = r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\archive (1)\chest_xray"
IMG_SIZE = (224, 224)

# --- 1. Load model ---
model = load_model("best_model_final.keras")

base_model_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model_layer = layer
        break

last_conv_layer_name = base_model_layer.layers[-1].name
print("Last conv layer:", last_conv_layer_name)


# --- 2. Grad-CAM heatmap generator ---
def make_gradcam_heatmap_v2(img_array, model, last_conv_layer_name, pred_index=None):
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

    with tf.GradientTape() as tape:
        conv_output, base_output = grad_model(img_array)
        tape.watch(conv_output)
        preds = head_model(base_output)
        if pred_index is None:
            pred_index = 0
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), preds.numpy()[0][0]


# --- 3. Preprocessing ---
def preprocess_single_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size, color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# --- 4. Overlay heatmap ---
def overlay_gradcam(img_path, heatmap, alpha=0.4, colormap="jet"):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (img.shape[0], img.shape[1]))
    heatmap_resized = tf.squeeze(heatmap_resized).numpy()

    cmap = plt.colormaps[colormap]
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]
    colored_heatmap = np.uint8(colored_heatmap * 255)

    superimposed = colored_heatmap * alpha + img * (1 - alpha)
    superimposed = np.uint8(superimposed)

    return img.astype(np.uint8), superimposed


# --- 5. Sanity check across multiple NORMAL + PNEUMONIA samples ---
def sanity_check_gradcam(base_dir, model, last_conv_layer_name, n_samples=4):
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 8))

    for row, cls in enumerate(["NORMAL", "PNEUMONIA"]):
        folder = os.path.join(base_dir, "test", cls)
        sample_files = random.sample(os.listdir(folder), n_samples)

        for col, fname in enumerate(sample_files):
            img_path = os.path.join(folder, fname)
            img_array = preprocess_single_image(img_path)
            heatmap, pred_prob = make_gradcam_heatmap_v2(img_array, model, last_conv_layer_name)
            _, overlay_img = overlay_gradcam(img_path, heatmap)

            axes[row, col].imshow(overlay_img)
            axes[row, col].set_title(f"{cls}\npred={pred_prob:.2f}", fontsize=10)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_sanity_check.png", dpi=150)
    plt.show()


sanity_check_gradcam(base_dir, model, last_conv_layer_name, n_samples=4)


# --- 6. Save individual Grad-CAM result (e.g. for app/demo use) ---
def save_gradcam_result(img_path, model, last_conv_layer_name, output_path="gradcam_output.png"):
    img_array = preprocess_single_image(img_path)
    heatmap, pred_prob = make_gradcam_heatmap_v2(img_array, model, last_conv_layer_name)
    original_img, overlay_img = overlay_gradcam(img_path, heatmap)

    risk_label = "High" if pred_prob >= 0.7 else "Medium" if pred_prob >= 0.3 else "Low"

    PILImage.fromarray(overlay_img).save(output_path)

    return {
        "probability": float(pred_prob),
        "risk_level": risk_label,
        "overlay_path": output_path
    }


sample_img_path = os.path.join(base_dir, "test", "PNEUMONIA", "person1_virus_6.jpeg")
result = save_gradcam_result(sample_img_path, model, last_conv_layer_name)
print(result)