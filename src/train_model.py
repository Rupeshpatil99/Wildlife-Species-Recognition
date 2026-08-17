import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 47

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/validation"


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

print("\nClasses:")
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
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


# Freeze pretrained layers

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
    "models/best_wildlife_model.keras",
    monitor="val_loss",
    save_best_only=True
)



model.summary()


EPOCHS = 20

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


print("\nTraining completed!")
