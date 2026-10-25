import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from NN import Convolution2D, MaxPooling2D, ReLU

np.random.seed(42)

def expected_dimensions(h, w, kernel, stride=1, padding=0):
    out_h = (h + 2 * padding - kernel) // stride + 1
    out_w = (w + 2 * padding - kernel) // stride + 1
    return out_h, out_w


def plot_filters_and_feature_maps(
    conv_layer, feature_maps, layer_name, output_dir
):
    """
    Plot the first three filters and their corresponding
    feature maps for one selected input image.
    """

    num_filters = min(3, conv_layer.output_channels)

    fig, axes = plt.subplots(
        num_filters, 2, figsize=(8, 3 * num_filters)
    )

    for i in range(num_filters):
        # Weight shape: (out_channels, in_channels, kh, kw)
        filter_weights = conv_layer.weights[i]

        # For a multi-channel filter, average across input channels
        # only for visualization.
        filter_image = filter_weights.mean(axis=0)

        axes[i, 0].imshow(filter_image, cmap="gray")
        axes[i, 0].set_title(f"Filter {i}")
        axes[i, 0].axis("off")

        # Feature maps are stored channel-first
        axes[i, 1].imshow(feature_maps[i], cmap="viridis")
        axes[i, 1].set_title(f"Feature map {i}")
        axes[i, 1].axis("off")

    fig.suptitle(f"{layer_name}: filters and feature maps")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{layer_name}_filters_and_maps.png"),
        dpi=200,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)
    
    

#layer 1
conv1 = Convolution2D(
    input_channels=1,
    output_channels=6,
    kernel_size_height=5,
    kernel_size_width=5
)
relu1 = ReLU()
max_pool1 = MaxPooling2D(
    pool_size=2,
    stride=2
)

#layer 2
conv2 = Convolution2D(
    input_channels=6,
    output_channels=16,
    kernel_size_height=5,
    kernel_size_width=5
)
relu2 = ReLU()
max_pool2 = MaxPooling2D(
    pool_size=2,
    stride=2
)

# Dataset and Q1 result paths
BaseDir_img = "../Group_16"

q1_csv_path = (
    "assignment_5/part_A/result_q1/"
    "feature_map_shapes_vs_expected_dimensions.csv"
)

image_name_data = pd.read_csv(q1_csv_path)

image_dict = {}

for _, row in image_name_data.iterrows():
    image_label = str(row["label"])
    image_name = str(row["image"])

    # Exact image selected in Q1:
    # Group_16/train/<label>/<image_name>
    image_path = os.path.join(
        BaseDir_img,
        "train",
        image_label,
        image_name
    )

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image from Q1 CSV not found: {image_path}"
        )

    with Image.open(image_path) as img:
        image = np.asarray(img.convert("L"), dtype=np.float64)

    # Verify against Q1's recorded input dimensions
    recorded_shape = tuple(
        int(v.strip())
        for v in row["input_shape"].strip("()").split(",")
    )

    if image.shape != recorded_shape:
        raise ValueError(
            f"Shape mismatch for {image_path}: "
            f"Q1 recorded {recorded_shape}, loaded {image.shape}"
        )

    # Required format: (channels, height, width)
    image = image[np.newaxis, :, :]

    image_key = f"{image_label}_{image_name}"
    image_dict[image_key] = image

print(f"Loaded {len(image_dict)} images selected in Q1:")
for image_key, image in image_dict.items():
    print(f"  {image_key}: {image.shape}")
    

output_dir = "result_q2"
os.makedirs(output_dir, exist_ok=True)

results = {}
shape_rows = []
    
for image_label, image in image_dict.items():

    # ----- Block 1 -----
    conv1_out = conv1.forward(image)
    relu1_out = relu1.forward(conv1_out)
    pool1_out = max_pool1.forward(relu1_out)

    # ----- Block 2 -----
    conv2_out = conv2.forward(pool1_out)
    relu2_out = relu2.forward(conv2_out)
    pool2_out = max_pool2.forward(relu2_out)

    results[image_label] = {
        "input": image,
        "conv1": conv1_out,
        "relu1": relu1_out,
        "pool1": pool1_out,
        "conv2": conv2_out,
        "relu2": relu2_out,
        "pool2": pool2_out,
    }

    # Print intermediate feature-map shapes
    print(f"\nImage: {image_label}")
    for name in ["conv1", "relu1", "pool1",
                 "conv2", "relu2", "pool2"]:
        print(f"{name}: {results[image_label][name].shape}")

    # Verify actual and expected dimensions
    h, w = image.shape[1:]
    expected_conv1 = expected_dimensions(h, w, kernel=5)
    expected_pool1 = expected_dimensions(*expected_conv1, kernel=2, stride=2)
    expected_conv2 = expected_dimensions(*expected_pool1, kernel=5)
    expected_pool2 = expected_dimensions(*expected_conv2, kernel=2, stride=2)

    stages = [
        ("conv1", conv1_out, (6, *expected_conv1)),
        ("pool1", pool1_out, (6, *expected_pool1)),
        ("conv2", conv2_out, (16, *expected_conv2)),
        ("pool2", pool2_out, (16, *expected_pool2)),
    ]

    for stage, output, expected_shape in stages:
        actual_shape = output.shape
        shape_rows.append({
            "image": image_label,
            "stage": stage,
            "actual_shape": str(actual_shape),
            "expected_shape": str(expected_shape),
            "dimensions_match": actual_shape == expected_shape,
        })

        assert actual_shape == expected_shape, (
            f"{image_label}, {stage}: "
            f"expected {expected_shape}, got {actual_shape}"
        )


output_dir = "result_q2"
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame(shape_rows).to_csv(
    os.path.join(
        output_dir,
        "feature_map_shapes_vs_expected_dimensions.csv"
    ),
    index=False
)


# Use the first selected image for the report figures
first_image_label = next(iter(results))

plot_filters_and_feature_maps(
    conv1,
    results[first_image_label]["conv1"],
    "conv1",
    output_dir
)

plot_filters_and_feature_maps(
    conv2,
    results[first_image_label]["conv2"],
    "conv2",
    output_dir
)