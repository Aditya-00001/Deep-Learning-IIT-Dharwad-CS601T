
from pathlib import Path
import zipfile
import numpy as np
import matplotlib.pyplot as plt

ZIP_PATH = Path("./train.zip") 
OUTPUT_DIR = Path("./handwriting_zero_centered_samples")
SAMPLES_PER_CLASS = 5
SEED = 42

CLASS_ORDER = ["a", "ai", "chA", "lA", "tA"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)


def load_class_samples(zip_file, class_name, count):
    candidates = [
        f"{name}" for name in zip_file.namelist()
        if name.endswith(".npy")
        and Path(name).parent.name == class_name
        and ("train" in Path(name).parts or len(Path(name).parts) >= 2)
    ]
    files = sorted(candidates)

    if len(files) < count:
        raise ValueError(
            f"Class {class_name} has {len(files)} .npy files in ZIP; "
            f"need at least {count}. Check ZIP_PATH/archive layout."
        )

    selected_idx = rng.choice(len(files), size=count, replace=False)
    selected_files = [files[i] for i in sorted(selected_idx)]

    samples = []
    for file_name in selected_files:
        with zip_file.open(file_name) as f:
            coords = np.load(f)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Expected centered coordinate array (T, 2) in {file_name}; "
                f"received {coords.shape}"
            )
        samples.append((Path(file_name).name, coords.astype(np.float32, copy=False)))
    return samples


def draw_trajectory(ax, coords, title):
    x = coords[:, 0]
    y = coords[:, 1]

    ax.plot(x, y, linewidth=1.25, marker=".", markersize=2.0)
    ax.scatter(x[0], y[0], s=18, marker="o", zorder=3, label="Start")

    radius = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), 1e-6)
    margin = 0.08 * radius
    ax.set_xlim(-radius - margin, radius + margin)
    ax.set_ylim(-radius - margin, radius + margin)
    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, linewidth=0.45, alpha=0.5)
    ax.axvline(0, linewidth=0.45, alpha=0.5)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="both", labelsize=5)
    ax.grid(False)


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"ZIP file not found: {ZIP_PATH}. Set ZIP_PATH to your train.zip."
        )

    samples_by_class = {}
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for class_name in CLASS_ORDER:
            samples_by_class[class_name] = load_class_samples(
                zf, class_name, SAMPLES_PER_CLASS
            )

    fig, axes = plt.subplots(
        len(CLASS_ORDER), SAMPLES_PER_CLASS,
        figsize=(8.0, 8.0), constrained_layout=True
    )

    for row, class_name in enumerate(CLASS_ORDER):
        for col, (filename, coords) in enumerate(samples_by_class[class_name]):
            draw_trajectory(axes[row, col], coords, f"Example {col + 1}")
            if col == 0:
                axes[row, col].set_ylabel(class_name, fontsize=9, fontweight="bold")

    fig.suptitle(
        "Zero-Centered Handwriting Trajectories: Five Samples per Class",
        fontsize=10, fontweight="bold"
    )
    plate = OUTPUT_DIR / "handwriting_zero_centered_5_each_5x5.png"
    fig.savefig(plate, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined 5x5 figure: {plate}")

    for class_name in CLASS_ORDER:
        fig, axes = plt.subplots(1, SAMPLES_PER_CLASS, figsize=(8.0, 2.2))
        for col, (filename, coords) in enumerate(samples_by_class[class_name]):
            draw_trajectory(axes[col], coords, f"Example {col + 1}")
        fig.suptitle(
            f"Class {class_name} — Zero-Centered Samples",
            fontsize=10, fontweight="bold"
        )
        fig.tight_layout()
        class_file = OUTPUT_DIR / f"class_{class_name}_zero_centered_5_samples.png"
        fig.savefig(class_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved class figure: {class_file}")

    manifest = OUTPUT_DIR / "sample_manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("Samples plotted directly from zero-centered .npy data; no normalization applied.\\n")
        for class_name in CLASS_ORDER:
            f.write(f"\\nClass {class_name}:\\n")
            for filename, _ in samples_by_class[class_name]:
                f.write(f"  {filename}\\n")
    print(f"Saved manifest: {manifest}")


if __name__ == "__main__":
    main()
