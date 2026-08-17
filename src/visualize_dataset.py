from pathlib import Path
import random

import matplotlib.pyplot as plt
from PIL import Image


DATA_DIR = Path("data/raw/animals/animals")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# Get all species folders
species_folders = [
    folder for folder in DATA_DIR.iterdir()
    if folder.is_dir()
]

# Select 9 random species
selected_species = random.sample(
    species_folders,
    min(9, len(species_folders))
)


plt.figure(figsize=(12, 10))


for i, species_folder in enumerate(selected_species):

    images = [
        image for image in species_folder.rglob("*")
        if image.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_path = random.choice(images)

    image = Image.open(image_path)

    plt.subplot(3, 3, i + 1)

    plt.imshow(image)

    plt.title(species_folder.name)

    plt.axis("off")


plt.tight_layout()

plt.show()