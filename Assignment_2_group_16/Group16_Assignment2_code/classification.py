import numpy as np
import os
from dl_utils import Data_preparation, Metrics, Plots
from NN import Dense, Logistic, FCNN, Trainer, SquaredError, Tanh


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "Classification/LS_Group16/LS_G16.csv"

NUM_CLASSES = 3

LEARNING_RATE = 0.01
MAX_EPOCHS = 1000
ERROR_THRESHOLD = 1e-4
CONVERGENCE_THRESHOLD = 1e-6

# Dataset 1 requires one hidden layer
ARCHITECTURES = {
    "2-3-3": [3],
    "2-5-3": [5],
    "2-8-3": [8],
    "2-10-3": [10],
}


# ============================================================
# Build FCNN
# ============================================================

def build_model(hidden_layers,activation="logistic"):

    layers = []

    input_size = 2

    for hidden_size in hidden_layers:

        layers.append(
            Dense(input_size, hidden_size)
        )

        layers.append(
            Logistic() if activation == "logistic" else Tanh()
        )

        input_size = hidden_size

    layers.append(
        Dense(input_size, NUM_CLASSES)
    )

    layers.append(
        Logistic() if activation == "logistic" else Tanh()
    )

    return FCNN(layers)


# ============================================================
# Run one architecture
# ============================================================

def run_experiment(
    architecture,
    X_train,
    Y_train,
    X_val,
    Y_val
):

    model = build_model(architecture, activation="logistic")

    loss = SquaredError()

    trainer = Trainer(
        model=model,
        loss_function=loss,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        error_threshold=ERROR_THRESHOLD,
        convergence_threshold=CONVERGENCE_THRESHOLD,
        validation_early_stopping=True
    )

    history = trainer.train(
        X_train,
        Y_train,
        X_val,
        Y_val
    )

    return model, history


# ============================================================
# Classification evaluation
# ============================================================

def evaluate_classification(
    model,
    X,
    Y
):

    output = model.predict(X)

    prediction = np.argmax(
        output,
        axis=1
    )

    metrics = Metrics(
        Y,
        prediction
    )

    return metrics, prediction


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs("results/classification/LS/logistics", exist_ok=True)
    prepare = Data_preparation()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = prepare.load_data(
        DATA_PATH
    )

    X = data[:, :-1]
    Y = data[:, -1].astype(int)

    # --------------------------------------------------------
    # Class-wise 60/20/20 split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test
    ) = prepare.stratified_train_val_test_split(
        X,
        Y,
        NUM_CLASSES
    )

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test
    ) = prepare.standardize(
        X_train,
        X_val,
        X_test
    )

    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    Y_train_encoded = prepare.one_hot_encode(
        Y_train,
        NUM_CLASSES, activation="logistic"
    )

    Y_val_encoded = prepare.one_hot_encode(
        Y_val,
        NUM_CLASSES, activation="logistic"
    )

    Y_test_encoded = prepare.one_hot_encode(
        Y_test,
        NUM_CLASSES, activation="logistic"
    )

    # --------------------------------------------------------
    # Architecture experiments
    # --------------------------------------------------------

    results = []

    for name, architecture in ARCHITECTURES.items():

        print("\n" + "=" * 60)
        print("Architecture:", name)
        print("=" * 60)

        model, history = run_experiment(
            architecture,
            X_train,
            Y_train_encoded,
            X_val,
            Y_val_encoded
        )

        # Validation predictions
        val_metrics, val_prediction = \
            evaluate_classification(
                model,
                X_val,
                Y_val
            )

        result = {
            "name": name,
            "architecture": architecture,
            "model": model,
            "history": history,
            "validation_metrics": val_metrics
        }

        results.append(result)

        # ----------------------------------------------------
        # Print validation results
        # ----------------------------------------------------

        print(
            "Best epoch:",
            history["best_epoch"]
        )

        print(
            "Best validation error:",
            history["best_validation_error"]
        )

        print(
            "Validation accuracy:",
            val_metrics.accuracy()
        )

        print(
            "Validation precision:",
            val_metrics.precision()
        )

        print(
            "Validation recall:",
            val_metrics.recall()
        )

        print(
            "Validation F1:",
            val_metrics.f1_score()
        )

        print(
            "Mean precision:",
            val_metrics.macro_precision()
        )

        print(
            "Mean recall:",
            val_metrics.macro_recall()
        )

        print(
            "Mean F1:",
            val_metrics.macro_f_score()
        )

    # ========================================================
    # Select best architecture
    # ========================================================

    best_result = min(
        results,
        key=lambda result:
        result["history"]["best_validation_error"]
    )

    best_model = best_result["model"]
    best_history = best_result["history"]

    print("\n")
    print("#" * 60)
    print("BEST ARCHITECTURE")
    print("#" * 60)

    print(
        "Architecture:",
        best_result["name"]
    )

    print(
        "Best epoch:",
        best_history["best_epoch"]
    )

    print(
        "Best validation error:",
        best_history["best_validation_error"]
    )

    # ========================================================
    # Final TEST evaluation
    # ========================================================

    test_metrics, test_prediction = \
        evaluate_classification(
            best_model,
            X_test,
            Y_test
        )

    print("\n")
    print("#" * 60)
    print("TEST RESULTS - BEST ARCHITECTURE")
    print("#" * 60)

    test_metrics.classification_report()

    # ========================================================
    # Plots
    # ========================================================

    plots = Plots()

    # --------------------------------------------------------
    # Error vs Epoch
    # --------------------------------------------------------

    # We will connect this to the exact history structure
    # returned by Trainer after confirming its keys.
    #
    plots.plot_error_history(
        best_history,
        save_path=(
            "results/classification/LS/logistics/"
            "best_error_vs_epoch.png"
        )
    )
    #
    # --------------------------------------------------------
    # Decision region
    # --------------------------------------------------------
    plots.plot_fcnn_decision_region(
        X_train,
        Y_train,
        best_model,
        save_path=(
            "results/classification/LS/logistics/"
            "best_decision_region.png"
        )
    )
    
    plots.plot_node_outputs_3d(
        X_train,
        Y_train,
        best_model,
        "Training",
        "results/classification/LS/logistics/node_outputs"
    )

    plots.plot_node_outputs_3d(
        X_val,
        Y_val,
        best_model,
        "Validation",
        "results/classification/LS/logistics/node_outputs"
    )

    plots.plot_node_outputs_3d(
        X_test,
        Y_test,
        best_model,
        "Test",
        "results/classification/LS/logistics/node_outputs"
    )
    # This should use best_model.predict() because our FCNN
    # returns raw output vectors and classification is done
    # using argmax.
    #
    # The current decision-region utility is designed for
    # a classifier returning class labels, so we will add/
    # adapt one generic FCNN decision-region method rather
    # than misuse the existing OVO method.
    #
    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    confusion = test_metrics.confusion_matrix()

    plots.plot_cm_from_scratch(
        confusion,
        labels=np.arange(NUM_CLASSES),
        save_path="results/classification/LS/logistics/best_test_confusion_matrix.png"
    )


if __name__ == "__main__":
    main()