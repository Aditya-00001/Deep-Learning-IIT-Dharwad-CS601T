"""Assignment 5 - Part A, Question 1: Manual 2D convolution.

Loads one image from each of the first five class folders, initializes one
Kaiming/He-normal 5x5 kernel, performs manual convolution, and saves the
kernel, feature maps, and dimension verification CSV.
"""

from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Reproducible kernel values.
np.random.seed(42)

# Adjust this path if your dataset is stored elsewhere.
BASE_DIR = Path("../../../Group_16/train")
OUTPUT_DIR = Path("result_q1")
KERNEL_SIZE = (5, 5)
STRIDE = 1
PADDING = 0


def initialize_kaiming_kernel(kernel_height, kernel_width, input_channels=1):
    """Create a Kaiming/He-normal kernel suitable for a following ReLU.

    For this single-channel manual 2D convolution, fan_in is kh * kw * channels.
    """
    fan_in = kernel_height * kernel_width * input_channels
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(kernel_height, kernel_width) * std


def manual_2D_convolution(input_matrix, kernel, stride=1, padding=0):
    """Apply a single-channel 2D convolution (cross-correlation) manually."""
    input_matrix = np.asarray(input_matrix, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    if input_matrix.ndim != 2 or kernel.ndim != 2:
        raise ValueError("input_matrix and kernel must both be 2D arrays")
    if stride < 1 or padding < 0:
        raise ValueError("stride must be >= 1 and padding must be >= 0")

    if padding:
        input_matrix = np.pad(
            input_matrix,
            ((padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0,
        )

    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    if input_height < kernel_height or input_width < kernel_width:
        raise ValueError("Kernel cannot be larger than the padded input")

    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1
    output_matrix = np.zeros((output_height, output_width), dtype=np.float64)

    for i in range(output_height):
        for j in range(output_width):
            start_i, start_j = i * stride, j * stride
            patch = input_matrix[
                start_i : start_i + kernel_height,
                start_j : start_j + kernel_width,
            ]
            output_matrix[i, j] = np.sum(patch * kernel)

    return output_matrix


def images_me(base_dir=BASE_DIR, number_of_classes=5):
    """Load the first image from each of the first classes, in sorted order."""
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {base_dir.resolve()}\n"
            "Update BASE_DIR at the top of Q1.py to point to your train folder."
        )

    image_dict = {}
    class_dirs = sorted(
        (p for p in base_dir.iterdir() if p.is_dir()), key=lambda p: p.name
    )[:number_of_classes]

    for class_dir in class_dirs:
        image_files = sorted(
            p for p in class_dir.iterdir() if p.is_file()
        )
        if not image_files:
            continue

        image_path = image_files[0]
        # MNIST images are grayscale; convert explicitly for consistent 2D input.
        with Image.open(image_path) as image:
            image_dict[f"{class_dir.name} {image_path.name}"] = np.asarray(
                image.convert("L")
            )

    if not image_dict:
        raise ValueError(f"No images found in class folders under {base_dir.resolve()}")
    return image_dict


def expected_dimensions(
    input_height, input_width, kernel_height, kernel_width, stride=1, padding=0
):
    """Calculate convolution output dimensions using the standard formula."""
    output_height = (input_height - kernel_height + 2 * padding) // stride + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride + 1
    return output_height, output_width


def run_q1(base_dir=BASE_DIR, output_dir=OUTPUT_DIR, show_plots=True):
    """Run Q1 and save all four requested artifacts; reusable with other paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dict = images_me(base_dir)
    kernel_height, kernel_width = KERNEL_SIZE
    kernel = initialize_kaiming_kernel(kernel_height, kernel_width, input_channels=1)

    # 1. Save the exact kernel values used for every image.
    pd.DataFrame(kernel).to_csv(
        output_dir / "kaiming_kernel_5x5.csv", index=False, header=False, float_format="%.10f"
    )

    # 2. Visualize and save the kernel.
    fig, ax = plt.subplots(figsize=(5, 4.5))
    image = ax.imshow(kernel, cmap="coolwarm", interpolation="nearest")
    ax.set_title("Kaiming/He-normal 5×5 Kernel")
    ax.set_xlabel("Kernel column")
    ax.set_ylabel("Kernel row")
    ax.set_xticks(range(kernel_width))
    ax.set_yticks(range(kernel_height))
    for row in range(kernel_height):
        for col in range(kernel_width):
            ax.text(col, row, f"{kernel[row, col]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Weight value")
    fig.tight_layout()
    fig.savefig(output_dir / "kaiming_kernel.png", dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    # 3. Apply the same kernel to one image per class and verify dimensions.
    feature_maps = {}
    shape_rows = []
    for label, img_array in image_dict.items():
        output = manual_2D_convolution(
            img_array, kernel, stride=STRIDE, padding=PADDING
        )
        expected = expected_dimensions(
            img_array.shape[0], img_array.shape[1], kernel_height, kernel_width,
            stride=STRIDE, padding=PADDING,
        )
        actual = output.shape
        feature_maps[label] = output
        shape_rows.append({
            "label": label.split()[0],
            "image": " ".join(label.split()[1:]),
            "input_shape": str(img_array.shape),
            "output_shape": str(actual),
            "expected_dimensions": str(expected),
            "dimensions_match": actual == expected,
        })
        print(f"{label}: output shape={actual}, expected={expected}, match={actual == expected}")

    # Feature maps: one panel per class.
    fig, axes = plt.subplots(1, len(feature_maps), figsize=(3 * len(feature_maps), 3.5))
    axes = np.atleast_1d(axes)
    for ax, (label, feature_map) in zip(axes, feature_maps.items()):
        ax.imshow(feature_map, cmap="gray")
        ax.set_title(label.split()[0])
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
        ax.axis("off")
    fig.suptitle("Manual convolution feature maps (5×5 Kaiming kernel)")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_maps.png", dpi=200, bbox_inches="tight")
    if show_plots:
        plt.show()
    plt.close(fig)

    pd.DataFrame(shape_rows).to_csv(
        output_dir / "feature_map_shapes_vs_expected_dimensions.csv", index=False
    )

    print(f"\nAll Q1 artifacts saved in: {output_dir.resolve()}")
    return kernel, feature_maps, pd.DataFrame(shape_rows)


if __name__ == "__main__":
    run_q1()
