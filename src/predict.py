import numpy as np
import tensorflow as tf



IMAGE_SIZE = (224, 224)

MODEL_PATH = "models/class_weighted_wildlife_model.keras"

IMAGE_PATH = "data/processed/test/lion/image_018.jpg"

TEST_DIR = "data/processed/test"



model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")



test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=32,
    shuffle=False
)

class_names = test_dataset.class_names

print("Number of classes:", len(class_names))


image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=IMAGE_SIZE
)



image_array = tf.keras.utils.img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)


predictions = model.predict(
    image_array,
    verbose=0
)


top_3_indices = np.argsort(
    predictions[0]
)[-3:][::-1]


print("\n" + "=" * 50)
print("TOP-3 WILDLIFE PREDICTIONS")
print("=" * 50)


for rank, index in enumerate(top_3_indices, start=1):

    species = class_names[index]

    confidence = predictions[0][index] * 100

    print(
        f"{rank}. {species:<15} {confidence:.2f}%"
    )


confidence = predictions[0][top_3_indices[0]] * 100


if confidence >= 70:

    status = "High Confidence"

elif confidence >= 40:

    status = "Medium Confidence"

else:

    status = "Low Confidence"


print("\nPrediction Status:", status)