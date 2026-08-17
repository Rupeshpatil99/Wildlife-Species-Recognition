from pathlib import Path
import random
import shutil


SOURCE_DIR = Path("data/raw/animals/animals")
OUTPUT_DIR = Path("data/processed")

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "validation"
TEST_DIR = OUTPUT_DIR / "test"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)


def get_images(folder):
    return [
        image
        for image in folder.iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
    ]


def create_split_directories(species):
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (split_dir / species).mkdir(
            parents=True,
            exist_ok=True
        )


def split_images(images):
    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    return train_images, val_images, test_images


def copy_images(images, destination):
    for image in images:
        shutil.copy2(
            image,
            destination / image.name
        )


def main():

    print("Preparing wildlife dataset...")

    species_folders = [
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]

    print(f"Found {len(species_folders)} species.")

    for species_folder in sorted(species_folders):

        species = species_folder.name

        images = get_images(species_folder)

        train_images, val_images, test_images = split_images(
            images
        )

        create_split_directories(species)

        copy_images(
            train_images,
            TRAIN_DIR / species
        )

        copy_images(
            val_images,
            VAL_DIR / species
        )

        copy_images(
            test_images,
            TEST_DIR / species
        )

        print(
            f"{species}: "
            f"train={len(train_images)}, "
            f"validation={len(val_images)}, "
            f"test={len(test_images)}"
        )

    print("\nDataset preparation completed.")


if __name__ == "__main__":
    main()