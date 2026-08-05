## ===== Overlay Heatmap on Original X-Ray =====
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)

# --- 1. Load the fine-tuned model ---
model = load_model("best_model_final.keras")

# --- 2. Get the nested DenseNet121 sub-model + last conv layer ---
base_model_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model_layer = layer
        break

last_conv_layer_name = base_model_layer.layers[-1].name  # "relu" for DenseNet121
print("Last conv layer:", last_conv_layer_name)


# --- 3. Grad-CAM heatmap generator ---
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


# --- 4. Preprocess a single image for the model ---
def preprocess_single_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size, color_mode="rgb")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# --- 5. Overlay heatmap on original full-res image ---
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


## ===== Run Grad-CAM on a sample test image =====
sample_img_path = r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\archive (1)\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg"

img_array = preprocess_single_image(sample_img_path)
heatmap, pred_prob = make_gradcam_heatmap_v2(img_array, model, last_conv_layer_name)
print(f"Predicted probability of PNEUMONIA: {pred_prob:.3f}")

original_img, overlay_img = overlay_gradcam(sample_img_path, heatmap)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(original_img.astype("uint8"))
axes[0].set_title("Original X-Ray")
axes[0].axis("off")

axes[1].imshow(heatmap, cmap="jet")
axes[1].set_title("Raw Grad-CAM Heatmap")
axes[1].axis("off")

axes[2].imshow(overlay_img)
axes[2].set_title(f"Overlay (Risk: {pred_prob:.1%})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("gradcam_overlay_sample.png", dpi=150)
plt.show()