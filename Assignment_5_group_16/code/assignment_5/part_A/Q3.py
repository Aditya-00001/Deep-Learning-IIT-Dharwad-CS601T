
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Package-safe imports: run with python -m assignment_5.part_A.Q3
try:
    from .NN.layers import (
        Convolution2D, MaxPooling2D, Flatten, Dense
    )
    from .NN.activations import ReLU, Softmax
    from .NN.losses import CrossEntropy
    from .NN.model import FCNN
    from .NN.trainer import Trainer
    from .metrics import evaluate_predictions, predict_labels
    from .q3_visualizations import (
        plot_conv_filters,
        plot_feature_maps,
        plot_top5_receptive_fields,
    )
except ImportError:
    # Fallback when executing from inside part_A.
    from NN.layers import (
        Convolution2D, MaxPooling2D, Flatten, Dense
    )
    from NN.activations import ReLU, Softmax
    from NN.losses import CrossEntropy
    from NN.model import FCNN
    from NN.trainer import Trainer
    from assignment_5.part_A.metrics import evaluate_predictions, predict_labels
    from assignment_5.part_A.q3_visualizations import (
        plot_conv_filters,
        plot_feature_maps,
        plot_top5_receptive_fields,
    )


# ============================================================
# Configuration
# ============================================================

SEED = 42
CLASSES = [1, 5, 6, 7, 9]
LABEL_TO_INDEX = {label: i for i, label in enumerate(CLASSES)}
INDEX_TO_LABEL = {i: label for label, i in LABEL_TO_INDEX.items()}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

RESULTS_DIR = SCRIPT_DIR / "results_q3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
USE_SAVED_MODEL = False

# Adjust this if your dataset is stored elsewhere.
DATASET_CANDIDATES = [
    PROJECT_ROOT / "Group_16",
    SCRIPT_DIR / "Group_16",
    Path.cwd() / "Group_16",
    Path.cwd().parent / "Group_16",
]

DATASET_ROOT = next(
    (p.resolve() for p in DATASET_CANDIDATES if p.is_dir()),
    None,
)

if DATASET_ROOT is None:
    raise FileNotFoundError(
        "Could not locate Group_16. Checked:\n"
        + "\n".join(str(p) for p in DATASET_CANDIDATES)
    )

np.random.seed(SEED)


# ============================================================
# Dataset loading
# ============================================================

def load_split(dataset_root, split):
    """Load one split as float images shaped (N, 1, 28, 28)."""
    split_dir = dataset_root / split
    images, labels = [], []

    for original_label in CLASSES:
        class_dir = split_dir / str(original_label)

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {
                ".png", ".jpg", ".jpeg", ".bmp"
            }:
                continue

            with Image.open(path) as im:
                im = im.convert("L").resize((28, 28))
                array = np.asarray(im, dtype=np.float32) / 255.0

            images.append(array[np.newaxis, :, :])
            labels.append(LABEL_TO_INDEX[original_label])

    if not images:
        raise ValueError(f"No images found in {split_dir}")

    return (
        np.stack(images).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def load_q1_reference_images(dataset_root):
    """
    Prefer the five reference images recorded in a Q1/Q2 CSV.
    The CSV must have columns named 'label' and 'image'.
    """
    search_root = PROJECT_ROOT
    candidates = []

    for csv_path in search_root.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if {"label", "image"}.issubset(df.columns):
            if len(df) == 5:
                try:
                    csv_labels = set(df["label"].astype(int))
                except (ValueError, TypeError):
                    continue

                if csv_labels == set(CLASSES):
                    candidates.append((csv_path, df))

    # Prefer the Q1/Q2 reference-image dimension CSV when available.
    candidates.sort(
        key=lambda item: (
            "feature_map_shapes_vs_expected_dimensions"
            not in item[0].name.lower(),
            "q1" not in item[0].name.lower(),
            str(item[0]),
        )
    )

    if candidates:
        csv_path, df = candidates[0]
        print(f"Reference-image CSV: {csv_path}")

        reference_images = []
        reference_names = []

        for _, row in df.iterrows():
            original_label = int(row["label"])
            filename = Path(str(row["image"])).name
            image_path = (
                dataset_root / "train" /
                str(original_label) / filename
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"CSV reference image not found: {image_path}"
                )

            with Image.open(image_path) as im:
                im = im.convert("L").resize((28, 28))
                array = np.asarray(im, dtype=np.float32) / 255.0

            reference_images.append(array[np.newaxis, :, :])
            reference_names.append(
                f"class_{original_label}_{filename}"
            )

        return np.stack(reference_images), reference_names

    # Explicit fallback: one sorted training image per class.
    # Prefer the CSV above so that Q1/Q2/Q3 use the same images.
    print(
        "WARNING: No five-row Q1/Q2 reference CSV found. "
        "Using the first sorted training image in each class."
    )

    reference_images, reference_names = [], []

    for original_label in CLASSES:
        class_dir = dataset_root / "train" / str(original_label)
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

        if not files:
            raise FileNotFoundError(
                f"No images available in {class_dir}"
            )

        path = files[0]
        with Image.open(path) as im:
            im = im.convert("L").resize((28, 28))
            array = np.asarray(im, dtype=np.float32) / 255.0

        reference_images.append(array[np.newaxis, :, :])
        reference_names.append(f"class_{original_label}_{path.name}")

    return np.stack(reference_images), reference_names


# ============================================================
# Model
# ============================================================

def build_model():
    return FCNN([
        Convolution2D(
            input_channels=1,
            output_channels=6,
            kernel_size_height=5,
            kernel_size_width=5,
            stride=1,
            padding=0,
        ),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2),

        Convolution2D(
            input_channels=6,
            output_channels=16,
            kernel_size_height=5,
            kernel_size_width=5,
            stride=1,
            padding=0,
        ),
        ReLU(),
        MaxPooling2D(pool_size=2, stride=2),

        Flatten(),
        Dense(256, 64),
        ReLU(),
        Dense(64, 5),
        Softmax(),
    ])


# ============================================================
# Evaluation and saving
# ============================================================

def save_confusion_matrix(matrix, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)

    positions = np.arange(len(CLASSES))
    ax.set_xticks(positions, labels=CLASSES)
    ax.set_yticks(positions, labels=CLASSES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center",
                color="black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(training_loss, validation_loss):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        np.arange(1, len(training_loss) + 1),
        training_loss,
        label="Training loss",
    )

    if validation_loss:
        ax.plot(
            np.arange(1, len(validation_loss) + 1),
            validation_loss,
            label="Validation loss",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Q3 Training and Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "loss_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_model_parameters(model):
    """Save learned parameters for later inspection or restoration."""
    arrays = {}

    for i, layer in enumerate(model.layers):
        if hasattr(layer, "weights"):
            arrays[f"layer_{i}_weights"] = layer.weights
        if hasattr(layer, "biases"):
            arrays[f"layer_{i}_biases"] = layer.biases

    np.savez(RESULTS_DIR / "model_parameters.npz", **arrays)


def load_model_parameters(model, path):
    """Load saved weights and biases into an identically built model."""
    saved = np.load(path)

    for i, layer in enumerate(model.layers):
        weights_key = f"layer_{i}_weights"
        biases_key = f"layer_{i}_biases"

        if weights_key in saved:
            layer.weights = saved[weights_key].copy()

        if biases_key in saved:
            layer.biases = saved[biases_key].copy()

    saved.close()


def evaluate_split(model, split_name, X, y):
    y_pred = predict_labels(model, X)
    result = evaluate_predictions(y, y_pred, num_classes=len(CLASSES))

    print(f"\n{'=' * 20} {split_name} {'=' * 20}")
    print(f"Accuracy:        {result['accuracy']:.4f}")
    print(f"Macro precision: {result['macro_precision']:.4f}")
    print(f"Macro recall:    {result['macro_recall']:.4f}")
    print(f"Macro F1:        {result['macro_f1']:.4f}")

    print("\nPer-class results:")
    for i, original_label in enumerate(CLASSES):
        print(
            f"Class {original_label}: "
            f"precision={result['precision_per_class'][i]:.4f}, "
            f"recall={result['recall_per_class'][i]:.4f}, "
            f"F1={result['f1_per_class'][i]:.4f}"
        )

    print("\nConfusion matrix (actual rows, predicted columns):")
    print(result["confusion_matrix"])

    save_confusion_matrix(
        result["confusion_matrix"],
        f"{split_name} confusion matrix",
        RESULTS_DIR / f"{split_name.lower()}_confusion_matrix.png",
    )

    # Save predictions so results can be examined later.
    pd.DataFrame({
        "true_encoded": y,
        "true_original_label": [INDEX_TO_LABEL[int(v)] for v in y],
        "predicted_encoded": y_pred,
        "predicted_original_label": [
            INDEX_TO_LABEL[int(v)] for v in y_pred
        ],
    }).to_csv(
        RESULTS_DIR / f"{split_name.lower()}_predictions.csv",
        index=False,
    )

    return result


# ============================================================
# Main training and visualization pipeline
# ============================================================

def main():
    print("Dataset root:", DATASET_ROOT)
    print("Results directory:", RESULTS_DIR)

    # 1. Load data
    X_train, y_train = load_split(DATASET_ROOT, "train")
    X_val, y_val = load_split(DATASET_ROOT, "val")
    X_test, y_test = load_split(DATASET_ROOT, "test")

    for name, X, y in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        print(
            f"{name:10s}: X={X.shape}, y={y.shape}, "
            f"encoded_classes={np.unique(y).tolist()}"
        )

    # One-hot targets are required by the current CrossEntropy setup.
    Y_train = np.eye(len(CLASSES), dtype=np.float64)[y_train]
    Y_val = np.eye(len(CLASSES), dtype=np.float64)[y_val]

    # 2. Load the same reference images used in Q1/Q2.
    q1_images, q1_names = load_q1_reference_images(DATASET_ROOT)
    print("Q1 reference images:", q1_names)

    # 3. Build model and verify a real image.
    model = build_model()
    sample_output = model.forward(X_train[0])

    print("\nSample output:", sample_output)
    print("Output shape:", sample_output.shape)
    print("Probability sum:", sample_output.sum())

    assert sample_output.shape == (5,)
    assert np.all(np.isfinite(sample_output))
    assert np.all(sample_output >= 0)
    assert np.isclose(sample_output.sum(), 1.0, atol=1e-7)

    print("Forward-pass check: PASSED")

    # 4. Train.
    # Start with a modest epoch count; increase after a successful run.
    
    saved_model_path = RESULTS_DIR / "model_parameters.npz"

    if USE_SAVED_MODEL and saved_model_path.exists():
        print("\nLoading existing one-epoch model; skipping training.")
        load_model_parameters(model, saved_model_path)

    else:
        trainer = Trainer(
            model=model,
            loss_function=CrossEntropy(),
            learning_rate=0.001,
            max_epochs=5,
            validation_early_stopping=True,
            patience=2,
            min_delta=1e-4,
            convergence_threshold=None,
        )

        start_time = time.time()
        history = trainer.train(X_train, Y_train, X_val, Y_val)
        elapsed = time.time() - start_time

        print(f"\nTraining time: {elapsed / 60:.2f} minutes")
        print(f"Best epoch: {history['best_epoch']}")
        print(
            "Best validation loss: "
            f"{history['best_validation_error']:.6f}"
        )

        pd.DataFrame({
            "epoch": np.arange(1, len(history["training_error"]) + 1),
            "training_loss": history["training_error"],
            "validation_loss": (
                history["validation_error"]
                + [np.nan] * (
                    len(history["training_error"])
                    - len(history["validation_error"])
                )
            ),
        }).to_csv(RESULTS_DIR / "training_history.csv", index=False)

        save_loss_curve(
            history["training_error"],
            history["validation_error"],
        )
        save_model_parameters(model)
        
    # 6. Evaluate all splits using the restored best checkpoint.
    # Test data is used only for final reporting, not model selection.
    metrics_summary = {}

    for split_name, X, y in [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        result = evaluate_split(model, split_name, X, y)
        metrics_summary[split_name] = {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in result.items()
        }
        metrics_summary[split_name]["confusion_matrix"] = (
            result["confusion_matrix"].tolist()
        )

    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # 7. Learned convolution filters.
    conv1 = model.layers[0]
    conv2 = model.layers[3]

    plot_conv_filters(
        conv1,
        "Conv1 learned filters (all 6)",
        max_filters=6,
        save_path=RESULTS_DIR / "conv1_filters.png",
    )
    plot_conv_filters(
        conv2,
        "Conv2 learned filters (10 of 16)",
        max_filters=10,
        save_path=RESULTS_DIR / "conv2_filters.png",
    )

    # 8. Feature maps and receptive fields for the same five Q1 images.
    for image, name in zip(q1_images, q1_names):
        safe_name = (
            name.replace("/", "_")
                .replace(" ", "_")
                .replace(".", "_")
        )
        output_prefix = str(RESULTS_DIR / safe_name)

        plot_feature_maps(
            model,
            image,
            image_title=name,
            output_prefix=output_prefix,
        )

        activation_values = plot_top5_receptive_fields(
            model,
            image,
            image_title=name,
            save_path=RESULTS_DIR / f"{safe_name}_receptive_fields.png",
        )

        print(
            f"Top-5 Conv2 activations for {name}: "
            f"{activation_values}"
        )

    print("\nQ3 pipeline finished.")
    print("All outputs saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
