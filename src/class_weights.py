from pathlib import Path
from collections import Counter

from sklearn.utils.class_weight import compute_class_weight
import numpy as np


TRAIN_DIR = Path("data/processed/train")


class_names = sorted([
    folder.name
    for folder in TRAIN_DIR.iterdir()
    if folder.is_dir()
])


class_counts = {}

for index, class_name in enumerate(class_names):

    image_count = len([
        image
        for image in (TRAIN_DIR / class_name).iterdir()
        if image.is_file()
    ])

    class_counts[index] = image_count


labels = []

for class_index, count in class_counts.items():

    labels.extend(
        [class_index] * count
    )


class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.array(labels)
)


class_weights = {
    index: weight
    for index, weight in enumerate(class_weights_array)
}


print("=" * 50)
print("CLASS WEIGHTS")
print("=" * 50)

for index, class_name in enumerate(class_names):

    print(
        f"{class_name:15s} "
        f"images={class_counts[index]:3d} "
        f"weight={class_weights[index]:.3f}"
    )