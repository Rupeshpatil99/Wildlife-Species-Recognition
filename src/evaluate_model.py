import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)




IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "data/processed/test"

MODEL_PATH = "models/class_weighted_wildlife_model.keras"




test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = test_dataset.class_names

print("\nNumber of classes:", len(class_names))

model = tf.keras.models.load_model(
    MODEL_PATH
)


print("\nModel loaded successfully!")


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

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_labels
    )
y_true = np.array(y_true)
y_pred = np.array(y_pred)



accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)



print("\n" + "=" * 50)
print("WILDLIFE SPECIES CLASSIFICATION RESULTS")
print("=" * 50)

print(f"\nAccuracy:          {accuracy:.4f}")
print(f"Accuracy (%):      {accuracy * 100:.2f}%")

print(f"\nMacro Precision:    {precision:.4f}")
print(f"Macro Recall:       {recall:.4f}")
print(f"Macro F1-score:     {macro_f1:.4f}")
print(f"Weighted F1-score:  {weighted_f1:.4f}")


print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)