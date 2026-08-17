import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import confusion_matrix


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "data/processed/test"
MODEL_PATH = "models/class_weighted_wildlife_model.keras"


# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names


# Load model
model = tf.keras.models.load_model(MODEL_PATH)


# Predictions
y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)


# Confusion matrix
cm = confusion_matrix(y_true, y_pred)


# Plot
plt.figure(figsize=(20, 18))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Wildlife Species Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

print("Confusion matrix saved successfully!")