import tensorflow as tf
import matplotlib.pyplot as plt


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/validation"
TEST_DIR = "data/processed/test"



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

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


class_names = train_dataset.class_names

print("\nNumber of classes:", len(class_names))

print("\nClasses:")
print(class_names)



data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


normalization = tf.keras.layers.Rescaling(1./255)



images, labels = next(iter(train_dataset))

print("\nOriginal image batch shape:")
print(images.shape)

print("\nOriginal pixel range:")
print(
    float(tf.reduce_min(images)),
    float(tf.reduce_max(images))
)


# Apply normalization
normalized_images = normalization(images)

print("\nNormalized pixel range:")
print(
    float(tf.reduce_min(normalized_images)),
    float(tf.reduce_max(normalized_images))
)



plt.figure(figsize=(10, 10))

for i in range(9):

    augmented_image = data_augmentation(
        tf.expand_dims(images[i], 0),
        training=True
    )

    plt.subplot(3, 3, i + 1)

    plt.imshow(
        augmented_image[0].numpy().astype("uint8")
    )

    plt.title(class_names[labels[i]])

    plt.axis("off")

plt.tight_layout()

plt.show()