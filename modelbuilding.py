## ===== PHASE 1: Data Preprocessing =====
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

base_dir = r"F:\KDU projects (extra)\Data odessey 2026-my proposals\XRAY\archive (1)\chest_xray"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=False,
    brightness_range=[0.9, 1.1]
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb"
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb"
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(base_dir, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    color_mode="rgb",
    shuffle=False
)

print("Class indices:", train_generator.class_indices)

train_labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)


## ===== PHASE 2: Build Model (Pretrained CNN with Frozen Base) =====
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False
print(f"Base model has {len(base_model.layers)} layers, all frozen for now")

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(1, activation="sigmoid")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()


## ===== PHASE 3: Stage 1 Training (frozen base, train head only) =====
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
    ModelCheckpoint("best_model_stage1.keras", monitor="val_auc", mode="max", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

history_stage1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=callbacks
)


## ===== PHASE 4: Stage 2 Fine-tuning (unfreeze top layers) =====
base_model.trainable = True

fine_tune_at = len(base_model.layers) - 30
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy",
             tf.keras.metrics.Precision(name="precision"),
             tf.keras.metrics.Recall(name="recall"),
             tf.keras.metrics.AUC(name="auc")]
)

callbacks_stage2 = [
    EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
    ModelCheckpoint("best_model_final.keras", monitor="val_auc", mode="max", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)
]

history_stage2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=callbacks_stage2
)


## ===== PHASE 5: Plot Training Curves =====
import matplotlib.pyplot as plt

def plot_history(h1, h2, metric):
    vals = h1.history[metric] + h2.history[metric]
    val_vals = h1.history[f"val_{metric}"] + h2.history[f"val_{metric}"]
    plt.plot(vals, label=f"train_{metric}")
    plt.plot(val_vals, label=f"val_{metric}")
    plt.axvline(x=len(h1.history[metric]), color="gray", linestyle="--", label="fine-tune start")
    plt.legend()
    plt.title(metric)
    plt.show()

plot_history(history_stage1, history_stage2, "auc")
plot_history(history_stage1, history_stage2, "recall")