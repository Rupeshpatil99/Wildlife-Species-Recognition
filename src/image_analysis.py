from pathlib import Path
from PIL import Image
from collections import Counter

DATA_DIR = Path("data/raw/animals/animals")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

species_counts = Counter()

total_images = 0
corrupted_images = []

image_sizes = Counter()

for species_folder in DATA_DIR.iterdir():

    if not species_folder.is_dir():
        continue

    for image_path in species_folder.rglob("*"):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        species_counts[species_folder.name] += 1
        total_images += 1

        try:
            with Image.open(image_path) as img:

                # Verify that the image can actually be opened
                img.verify()

            # Open again to get image dimensions
            with Image.open(image_path) as img:
                image_sizes[img.size] += 1

        except Exception:
            corrupted_images.append(str(image_path))


print("=" * 50)
print("WILDLIFE DATASET ANALYSIS")
print("=" * 50)

print(f"\nNumber of species: {len(species_counts)}")
print(f"Total images: {total_images}")

print("\nImages per species:")
for species, count in sorted(species_counts.items()):
    print(f"{species}: {count}")

print("\nCorrupted images:", len(corrupted_images))

if corrupted_images:
    print("\nCorrupted image files:")
    for image in corrupted_images:
        print(image)

print("\nMost common image dimensions:")

for size, count in image_sizes.most_common(10):
    print(f"{size}: {count} images")