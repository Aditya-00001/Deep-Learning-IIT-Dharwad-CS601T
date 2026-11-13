# preprocessing.py

import re
import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = Path("../data/dataset_")

HANDWRITING_DIR = BASE_DIR / "dataset_hadnwriting"
CV_DIR = BASE_DIR / "dataset_cv_sound"

OUTPUT_DIR = Path("../data/dataset_prepared_centered")

HANDWRITING_OUTPUT = OUTPUT_DIR / "handwriting"
CV_OUTPUT = OUTPUT_DIR / "cv_speech"

VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

IMAGES_PER_CLASS = 1

RANDOM_SEED = 42

def natural_sort_key(path):
    """
    Sort filenames naturally:
    1.txt, 2.txt, 10.txt instead of 1.txt, 10.txt, 2.txt
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def get_class_folders(root_dir):
    if not root_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {root_dir.resolve()}"
        )

    class_folders = sorted(
        [folder for folder in root_dir.iterdir() if folder.is_dir()],
        key=lambda p: p.name
    )

    if not class_folders:
        raise ValueError(f"No class folders found inside {root_dir}")

    return class_folders


def get_data_files(folder):
    if not folder.exists():
        raise FileNotFoundError(
            f"Data folder does not exist: {folder.resolve()}"
        )

    files = [
        file for file in folder.iterdir()
        if file.is_file()
    ]

    return sorted(files, key=natural_sort_key)


def save_json(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def extract_numbers(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    number_pattern = (
        r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    )

    values = re.findall(number_pattern, content)

    return np.asarray(values, dtype=np.float64)


def read_handwriting_file(filepath):

    values = extract_numbers(filepath)

    if len(values) == 0:
        raise ValueError(f"Empty file: {filepath}")

    num_points = int(values[0])

    coordinates = values[1:]

    expected_values = num_points * 2

    if len(coordinates) != expected_values:
        raise ValueError(
            f"Invalid handwriting file: {filepath}\n"
            f"Expected {expected_values} coordinate values, "
            f"got {len(coordinates)}"
        )

    coordinates = coordinates.reshape(num_points, 2)

    return coordinates

# def standardize_handwriting(coordinates):

#     coordinates = coordinates.astype(np.float32)

#     standardized = coordinates.copy()

#     for axis in range(2):

#         mean = np.mean(coordinates[:, axis])

#         std = np.std(coordinates[:, axis])

#         if std < 1e-8:
#             standardized[:, axis] = 0.0

#         else:
#             standardized[:, axis] = (
#                 coordinates[:, axis] - mean
#             ) / std

#     return standardized.astype(np.float32)

def center_handwriting(coordinates):

    coordinates = coordinates.astype(np.float32)

    mean_x = np.mean(coordinates[:, 0])
    mean_y = np.mean(coordinates[:, 1])

    centered_coordinates = coordinates.copy()

    centered_coordinates[:, 0] -= mean_x
    centered_coordinates[:, 1] -= mean_y

    return centered_coordinates


# def normalize_handwriting(coordinates):

#     normalized = coordinates.copy()

#     for axis in range(2):

#         minimum = np.min(coordinates[:, axis])
#         maximum = np.max(coordinates[:, axis])

#         if maximum - minimum == 0:
#             normalized[:, axis] = 0.0

#         else:
#             normalized[:, axis] = (
#                 coordinates[:, axis] - minimum
#             ) / (maximum - minimum)

#     return normalized.astype(np.float32)


def plot_handwriting(
    coordinates,
    class_name,
    sample_name,
    save_path,
    title
):

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    plt.figure(figsize=(5, 5))

    plt.plot(
        x,
        y,
        marker=".",
        markersize=3,
        linewidth=1.5
    )

    plt.scatter(
        x[0],
        y[0],
        color="green",
        s=50,
        label="Start"
    )

    plt.scatter(
        x[-1],
        y[-1],
        color="red",
        s=50,
        label="End"
    )

    plt.title(
        f"{title}\nClass: {class_name} | Sample: {sample_name}"
    )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    if title == "Normalized Handwriting":
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def prepare_handwriting_dataset():

    print("\n" + "=" * 60)
    print("PREPARING HANDWRITING DATASET")
    print("=" * 60)

    class_folders = get_class_folders(HANDWRITING_DIR)

    class_mapping = {
        class_folder.name: index
        for index, class_folder in enumerate(class_folders)
    }

    save_json(
        class_mapping,
        HANDWRITING_OUTPUT / "class_mapping.json"
    )

    print("\nClass Mapping:")
    print(class_mapping)

    manifest = []

    total_train = 0
    total_test = 0

    for class_folder in class_folders:

        class_name = class_folder.name
        class_label = class_mapping[class_name]

        train_dir = class_folder / "train"
        test_dir = class_folder / "dev"

        train_files = get_data_files(train_dir)
        test_files = get_data_files(test_dir)

        print(f"\nProcessing class: {class_name}")
        print(f"Train samples: {len(train_files)}")
        print(f"Test samples:  {len(test_files)}")

        total_train += len(train_files)
        total_test += len(test_files)

        for split, files in [
            ("train", train_files),
            ("test", test_files)
        ]:

            for filepath in files:

                try:

                    original_coordinates = read_handwriting_file(
                        filepath
                    )

                    center_coordinates = center_handwriting(
                        original_coordinates
                    )
                    output_dir = (
                        HANDWRITING_OUTPUT
                        / split
                        / class_name
                    )

                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    output_file = (
                        output_dir
                        / f"{filepath.stem}.npy"
                    )

                    np.save(
                        output_file,
                        center_coordinates
                    )

                    manifest.append({
                        "sample_id": filepath.name,
                        "class_name": class_name,
                        "label": class_label,
                        "split": split,
                        "sequence_length": len(
                            center_coordinates
                        ),
                        "feature_dim": 2,
                        "processed_path": str(
                            output_file.as_posix()
                        )
                    })

                except Exception as e:

                    print(
                        f"ERROR processing {filepath}: {e}"
                    )

        plot_files = train_files[:IMAGES_PER_CLASS]

        for index, filepath in enumerate(plot_files):

            original_coordinates = read_handwriting_file(
                filepath
            )

            center_coordinates = center_handwriting(
                original_coordinates
            )

            original_plot_path = (
                VISUALIZATION_DIR
                / "handwriting"
                / "original"
                / class_name
                / f"{filepath.stem}_original.png"
            )

            plot_handwriting(
                coordinates=original_coordinates,
                class_name=class_name,
                sample_name=filepath.name,
                save_path=original_plot_path,
                title="Original Handwriting"
            )

            normalized_plot_path = (
                VISUALIZATION_DIR
                / "handwriting"
                / "normalized"
                / class_name
                / f"{filepath.stem}_centered.png"
            )

            plot_handwriting(
                coordinates=center_coordinates,
                class_name=class_name,
                sample_name=filepath.name,
                save_path=normalized_plot_path,
                title="Centered Handwriting"
            )


    manifest_path = (
        HANDWRITING_OUTPUT / "sample_manifest.csv"
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if manifest:

        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=manifest[0].keys()
            )

            writer.writeheader()
            writer.writerows(manifest)

    print("\nHandwriting Dataset Summary")
    print("-" * 40)
    print(f"Number of classes: {len(class_mapping)}")
    print(f"Training samples:  {total_train}")
    print(f"Test samples:      {total_test}")
    print(
        "Processed sequences:",
        len(manifest)
    )

    print("\nHandwriting preprocessing completed!")

    return class_mapping


def read_mfcc_file(filepath):

    values = extract_numbers(filepath)

    if len(values) == 0:
        raise ValueError(f"Empty MFCC file: {filepath}")

    feature_dim = 39

    if len(values) % feature_dim != 0:
        raise ValueError(
            f"Invalid MFCC file: {filepath}\n"
            f"Number of values ({len(values)}) "
            f"is not divisible by 39."
        )

    mfcc = values.reshape(-1, feature_dim)

    return mfcc


def get_cv_files(folder):

    files = get_data_files(folder)

    valid_extensions = {
        ".mfcc",
        ".txt",
        ""
    }

    files = [
        file for file in files
        if file.suffix.lower() in valid_extensions
    ]

    return files


def prepare_cv_dataset():

    print("\n" + "=" * 60)
    print("PREPARING CV SPEECH DATASET")
    print("=" * 60)

    class_folders = get_class_folders(CV_DIR)

    class_mapping = {
        class_folder.name: index
        for index, class_folder in enumerate(class_folders)
    }

    save_json(
        class_mapping,
        CV_OUTPUT / "class_mapping.json"
    )

    print("\nClass Mapping:")
    print(class_mapping)

    train_samples = []
    test_samples = []

    for class_folder in class_folders:

        class_name = class_folder.name
        class_label = class_mapping[class_name]

        train_dir = class_folder / "Train"
        test_dir = class_folder / "Test"

        train_files = get_cv_files(train_dir)
        test_files = get_cv_files(test_dir)

        print(f"\nProcessing class: {class_name}")
        print(f"Train samples: {len(train_files)}")
        print(f"Test samples:  {len(test_files)}")

        for filepath in train_files:

            try:

                mfcc = read_mfcc_file(filepath)

                train_samples.append({
                    "sample_id": filepath.name,
                    "class_name": class_name,
                    "label": class_label,
                    "sequence": mfcc,
                    "split": "train"
                })

            except Exception as e:

                print(
                    f"ERROR processing {filepath}: {e}"
                )

        for filepath in test_files:

            try:

                mfcc = read_mfcc_file(filepath)

                test_samples.append({
                    "sample_id": filepath.name,
                    "class_name": class_name,
                    "label": class_label,
                    "sequence": mfcc,
                    "split": "test"
                })

            except Exception as e:

                print(
                    f"ERROR processing {filepath}: {e}"
                )

    if not train_samples:
        raise ValueError("No valid CV training samples found.")

    print("\nCalculating MFCC training mean and standard deviation...")

    manifest = []

    for samples in [
        train_samples,
        test_samples
    ]:

        for sample in samples:

            mfcc = sample["sequence"]
            
            processed_mfcc = mfcc.astype(np.float32)
            
            split = sample["split"]
            class_name = sample["class_name"]

            output_dir = (
                CV_OUTPUT
                / split
                / class_name
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_file = (
                output_dir
                / f"{Path(sample['sample_id']).stem}.npy"
            )

            np.save(
                output_file,
                processed_mfcc
            )

            manifest.append({
                "sample_id": sample["sample_id"],
                "class_name": class_name,
                "label": sample["label"],
                "split": split,
                "sequence_length": len(
                    processed_mfcc
                ),
                "feature_dim": 39,
                "processed_path": str(
                    output_file.as_posix()
                )
            })


    manifest_path = (
        CV_OUTPUT / "sample_manifest.csv"
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if manifest:

        with open(
            manifest_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=manifest[0].keys()
            )

            writer.writeheader()
            writer.writerows(manifest)

    print("\nCV Speech Dataset Summary")
    print("-" * 40)
    print(f"Number of classes: {len(class_mapping)}")
    print(f"Training samples:  {len(train_samples)}")
    print(f"Test samples:      {len(test_samples)}")
    print(f"Processed samples: {len(manifest)}")
    print(f"Feature dimension: 39")

    print("\nCV preprocessing completed!")

    return class_mapping

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    handwriting_mapping = prepare_handwriting_dataset()

    cv_mapping = prepare_cv_dataset()

    summary = {
        "handwriting_classes": handwriting_mapping,
        "cv_classes": cv_mapping,
        "handwriting_feature_dim": 2,
        "cv_feature_dim": 39,
        "handwriting_normalization": "Per-sample min-max normalization",
        "cv_normalization": "Training-set feature-wise standardization"
    }

    save_json(
        summary,
        OUTPUT_DIR / "dataset_summary.json"
    )

    print("\n" + "=" * 60)
    print("ALL DATASETS PREPARED SUCCESSFULLY!")
    print("=" * 60)

    print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
