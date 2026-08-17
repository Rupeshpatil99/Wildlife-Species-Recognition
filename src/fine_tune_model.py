import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models



IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 47

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/validation"

BASE_MODEL_PATH = "models/best_wildlife_model.keras"
FINE_TUNED_MODEL_PATH = "models/fine_tuned_wildlife_model.keras"



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

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    AUTOTUNE
)


model = tf.keras.models.load_model(
    BASE_MODEL_PATH
)

print("\nBaseline model loaded.")


base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        base_model = layer
        break


if base_model is None:

    raise ValueError(
        "MobileNetV2 base model could not be found."
    )


print(
    "\nBase model:",
    base_model.name
)


base_model.trainable = True


# Freeze earlier layers
# and fine-tune only later layers

fine_tune_at = 100


for layer in base_model.layers[:fine_tune_at]:

    layer.trainable = False



model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    FINE_TUNED_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True
)



EPOCHS = 15

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


print("\nFine-tuning completed!")

print(
    "\nBest fine-tuned model saved to:"
)

print(
    FINE_TUNED_MODEL_PATH
)