import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
np.random.seed(42)  # For reproducibility

# Initialize Kaiming (He) Normal Kernel
def initialize_kaiming_kernel(kernel_height, kernel_width, input_channels=1):
    """
    Generates a kernel initialized with Kaiming (He) Normal initialization.
    Suitable for layers followed by ReLU activations.
    """
    # Calculate fan-in (number of input units)
    fan_in = kernel_height * kernel_width * input_channels
    
    # Calculate standard deviation for ReLU: sqrt(2 / fan_in)
    std = np.sqrt(2.0 / fan_in)
    
    # Sample from a standard normal distribution and scale
    kernel = np.random.randn(kernel_height, kernel_width) * std
    
    return kernel

# Manual 2D Convolution Implementation
def manual_2D_convolution(input_matrix, kernel, stride=1, padding=0):
    np_input_matrix = np.array(input_matrix)
    
    if padding > 0:
        np_input_matrix = np.pad(np_input_matrix, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
    input_height, input_width = np_input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            start_i = i * stride
            start_j = j * stride
            output_matrix[i,j] = np.sum(np_input_matrix[start_i:start_i+kernel_height, start_j:start_j+kernel_width] * kernel)
    
    return output_matrix

# Function to load and display the first image from each of the first 5 classes
def images_me():
    # Define the main directory path
    base_dir = "../../../Group_16/train"

    # Dictionary to store the image arrays
    image_dict = {}

    # Get all subdirectories (classes) and select the first 5
    classes = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ][:5]

    for label in classes:
        class_path = os.path.join(base_dir, label)

        # List all files in the class subdirectory
        files = [
            f
            for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ]

        if files:
            # Pick the first image from the class
            first_img_name = files[0]
            img_path = os.path.join(class_path, first_img_name)

            # Open image with Pillow, convert to NumPy array, and store in dictionary
            with Image.open(img_path) as img:
                image_dict[f"{label} {first_img_name}"] = np.array(img)
    # # Optional: Verify and plot the images using Matplotlib
    # for i, (label, img_array) in enumerate(image_dict.items()):
    #     plt.subplot(1, 5, i + 1)
    #     plt.imshow(img_array, cmap='gray')
    #     plt.title(label)
    #     plt.axis("off")

    # plt.show()
    
    return image_dict

def expected_dimensions(input_height, input_width, kernel_height, kernel_width, stride=1, padding=0):
    """
    Calculate the expected output dimensions after applying a 2D convolution.
    """
    output_height = (input_height - kernel_height + 2 * padding) // stride + 1
    output_width = (input_width - kernel_width + 2 * padding) // stride + 1
    return output_height, output_width

image_dict = images_me()
kernel = initialize_kaiming_kernel(5,5, input_channels=1)
feature_map = {label: {} for label in image_dict.keys()}
for label, img_array in image_dict.items():
    output = manual_2D_convolution(img_array, kernel)
    feature_map[label]["output"] = output
    feature_map[label]["expected_dimensions"] = expected_dimensions(img_array.shape[0], img_array.shape[1], kernel.shape[0], kernel.shape[1])

for i, (label, feature) in enumerate(feature_map.items()):
    plt.subplot(1, 5, i + 1)
    plt.imshow(feature["output"], cmap='gray')
    plt.title(label.split()[0])  # Display only the class label
    plt.axis("off")
plt.savefig("result_q1/feature_maps.png")
plt.show()

#shapes
for label, feature in feature_map.items():
    print(f"Feature map for {label}: {feature['output'].shape} {feature['expected_dimensions']}")
    
df = pd.DataFrame.from_dict({label.split()[0]: {"output_shape": feature["output"].shape, "expected_dimensions": feature["expected_dimensions"]} for label, feature in feature_map.items()}, orient='index', 
                            columns=["output_shape", "expected_dimensions"])
df.to_csv("result_q1/feature_map_shapes_vs_expected_dimensions.csv", index=True)