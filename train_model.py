import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
import os

# -----------------------------
# 1. Basic settings
# -----------------------------
DATASET_PATH = "dataset"
IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

# -----------------------------
# 2. Load dataset
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nFood Classes:")
for i, name in enumerate(class_names):
    print(i, name)

print("\nNumber of classes:", num_classes)

# Save class names
os.makedirs("model", exist_ok=True)

with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

# -----------------------------
# 3. Improve dataset performance
# -----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# 4. Data augmentation
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# -----------------------------
# 5. Pretrained MobileNetV2
# -----------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# -----------------------------
# 6. Build our ML model
# -----------------------------
inputs = layers.Input(shape=(160, 160, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# -----------------------------
# 7. Compile model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 8. Train model
# -----------------------------
print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# 9. Save trained model
# -----------------------------
model.save("model/food_model.keras")

print("\nModel saved successfully!")
print("Location: model/food_model.keras")

# -----------------------------
# 10. Plot accuracy
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("model/accuracy.png")
plt.close()

# -----------------------------
# 11. Plot loss
# -----------------------------
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("model/loss.png")
plt.close()

print("Graphs saved in model/ folder.")
print("\nTraining completed!")