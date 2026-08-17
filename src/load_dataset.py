import tensorflow as tf

# Dataset paths
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/validation"
TEST_DIR = "data/processed/test"

# Image configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Get class names
class_names = train_dataset.class_names

print("\nNumber of classes:", len(class_names))
print("\nClasses:")
print(class_names)

print("\nDataset loading completed!")