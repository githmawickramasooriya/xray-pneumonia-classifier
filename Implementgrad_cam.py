## ===== PHASE 7: Grad-CAM Implementation =====
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

base_dir = r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\archive (1)\chest_xray"
IMG_SIZE = (224, 224)

# --- 1. Load the fine-tuned model ---
model = load_model("best_model_final.keras")
model.summary()

# --- 2. Get the nested DenseNet121 sub-model ---
base_model_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model_layer = layer
        break

# Use the base model's own final layer (standard for DenseNet121 Grad-CAM)
last_conv_layer_name = base_model_layer.layers[-1].name
print("Last conv layer:", last_conv_layer_name)


# --- 3. Grad-CAM heatmap function ---
def make_gradcam_heatmap(img_array, model, base_model_layer, last_conv_layer_name, pred_index=None):
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


# --- 4. Load and preprocess a sample test image ---
sample_dir = os.path.join(base_dir, "test", "PNEUMONIA")
sample_filename = os.listdir(sample_dir)[0]
sample_path = os.path.join(sample_dir, sample_filename)

img = image.load_img(sample_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array_batch = np.expand_dims(img_array, axis=0)

# --- 5. Generate Grad-CAM heatmap ---
heatmap, prediction = make_gradcam_heatmap(
    img_array_batch, model, base_model_layer, last_conv_layer_name
)
print(f"File: {sample_filename} | Predicted probability (PNEUMONIA): {prediction:.3f}")


# --- 6. Overlay heatmap on original image ---
heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], IMG_SIZE).numpy().squeeze()

jet = plt.colormaps["jet"]          # <-- fixed line
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[np.uint8(255 * heatmap_resized)]

superimposed = jet_heatmap * 0.4 + img_array
superimposed = np.clip(superimposed, 0, 1)

# --- 7. Display side-by-side: original vs Grad-CAM overlay ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img_array)
axes[0].set_title(f"Original\n({sample_filename})")
axes[0].axis("off")

axes[1].imshow(superimposed)
axes[1].set_title(f"Grad-CAM\nPred: {prediction:.3f}")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("gradcam_sample.png")
plt.show()