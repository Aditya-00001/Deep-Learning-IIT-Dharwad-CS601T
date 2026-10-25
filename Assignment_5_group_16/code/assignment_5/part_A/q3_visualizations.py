
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_conv_filters(conv_layer, title, max_filters=10, save_path=None):
    weights = conv_layer.weights
    count = min(max_filters, weights.shape[0])

    fig, axes = plt.subplots(
        1, count, figsize=(2 * count, 2.5), squeeze=False
    )

    for i in range(count):
        # Shape: (input_channels, kernel_height, kernel_width)
        kernel = weights[i]
        displayed_kernel = kernel[0] if kernel.shape[0] == 1 else kernel.mean(axis=0)

        axes[0, i].imshow(displayed_kernel, cmap="gray")
        axes[0, i].set_title(f"Filter {i}")
        axes[0, i].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def extract_conv_activations(model, image):
    """Return post-ReLU activations from Conv1 and Conv2."""
    x = image

    conv1_activation = None
    conv2_activation = None

    for i, layer in enumerate(model.layers):
        x = layer.forward(x)

        if i == 1:       # Conv1 -> ReLU
            conv1_activation = x.copy()

        elif i == 4:     # Conv2 -> ReLU
            conv2_activation = x.copy()

    return conv1_activation, conv2_activation


def plot_feature_maps(model, image, image_title, output_prefix):
    conv1_maps, conv2_maps = extract_conv_activations(model, image)

    # Plot every Conv1 channel and the first 10 Conv2 channels.
    for layer_name, maps, count in [
        ("conv1", conv1_maps, min(6, conv1_maps.shape[0])),
        ("conv2", conv2_maps, min(10, conv2_maps.shape[0])),
    ]:
        fig, axes = plt.subplots(
            1, count, figsize=(2 * count, 2.5), squeeze=False
        )

        for channel in range(count):
            axes[0, channel].imshow(maps[channel], cmap="viridis")
            axes[0, channel].set_title(f"Map {channel}")
            axes[0, channel].axis("off")

        fig.suptitle(f"{image_title} — {layer_name} feature maps")
        fig.tight_layout()
        fig.savefig(
            f"{output_prefix}_{layer_name}_maps.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

    return conv1_maps, conv2_maps


def plot_top5_receptive_fields(model, image, image_title, save_path):
    """
    Find the five strongest Conv2 post-ReLU activations and
    map their theoretical receptive fields onto the input image.
    """
    _, conv2_maps = extract_conv_activations(model, image)

    # Conv2 activation tensor: (channels, height, width)
    channels, height, width = conv2_maps.shape

    flat_indices = np.argsort(conv2_maps.ravel())[::-1][:5]
    top_values = []

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image[0], cmap="gray")

    for rank, flat_index in enumerate(flat_indices, start=1):
        channel, row, col = np.unravel_index(
            flat_index, conv2_maps.shape
        )
        activation = float(conv2_maps[channel, row, col])
        top_values.append(activation)

        # Theoretical RF: 14 x 14 pixels; effective stride = 2.
        x0 = 2 * col
        y0 = 2 * row
        rf_size = 14

        # Clip the rectangle to the 28 x 28 image.
        x1 = min(x0 + rf_size, image.shape[2])
        y1 = min(y0 + rf_size, image.shape[1])

        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
        ax.add_patch(rect)

        ax.text(
            x0,
            max(0, y0 - 0.5),
            f"{rank}: F{channel}",
            color="yellow",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.7, pad=1),
        )

    ax.set_title(f"Top-5 Conv2 activations — {image_title}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return top_values