
import numpy as np


def evaluate_predictions(y_true, y_pred, num_classes=5):
    """Return accuracy, confusion matrix, and per-class metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    confusion = np.zeros((num_classes, num_classes), dtype=int)

    for actual, predicted in zip(y_true, y_pred):
        confusion[actual, predicted] += 1

    accuracy = np.mean(y_true == y_pred)

    tp = np.diag(confusion).astype(float)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    precision = np.divide(
        tp, tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) != 0,
    )
    recall = np.divide(
        tp, tp + fn,
        out=np.zeros_like(tp),
        where=(tp + fn) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": confusion,
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }


def predict_labels(model, X):
    predictions = []

    for image in X:
        probabilities = model.forward(image)
        predictions.append(int(np.argmax(probabilities)))

    return np.asarray(predictions, dtype=int)
