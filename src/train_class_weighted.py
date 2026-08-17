import tensorflow as tf
import numpy as np

from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers
from tensorflow.keras import models


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 47

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/validation"

MODEL_PATH = "models/class_weighted_wildlife_model.keras"



train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_dataset.class_names

print("\nNumber of classes:", len(class_names))



AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)


class_counts = []

for class_name in class_names:

    class_folder = Path(TRAIN_DIR) / class_name

    image_count = len([
        image
        for image in class_folder.iterdir()
        if image.is_file()
    ])

    class_counts.append(image_count)


labels = []

for class_index, count in enumerate(class_counts):

    labels.extend(
        [class_index] * count
    )


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=np.array(labels)
)


class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights_array)
}


print("\nClass weights:")

for index, class_name in enumerate(class_names):

    print(
        f"{class_name:15s} "
        f"weight={class_weights[index]:.3f}"
    )


data_augmentation = models.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.1),

    layers.RandomZoom(0.1)

])



base_model = tf.keras.applications.MobileNetV2(

    input_shape=(224, 224, 3),

    include_top=False,

    weights="imagenet"
)


base_model.trainable = False



inputs = layers.Input(
    shape=(224, 224, 3)
)


x = data_augmentation(inputs)

x = layers.Rescaling(1.0 / 255)(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)


EPOCHS = 20

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


print("\nClass-weighted training completed!")

print(
    f"\nModel saved to: {MODEL_PATH}"
)