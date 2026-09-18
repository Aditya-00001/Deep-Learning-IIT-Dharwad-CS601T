import os
import numpy as np

from dl_utils import Data_preparation, Plots, Metrics
from NN import Dense, Tanh, Linear, FCNN, Trainer, SquaredError


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Change ONLY the dataset path
#
# Univariate:
# Regression/UnivariateData/16.csv
#
# Bivariate:
# Regression/BivariateData/16.csv
# ------------------------------------------------------------

DATA_PATH = "Regression/BivariateData/16.csv"


LEARNING_RATE = 0.001
MAX_EPOCHS = 1000

ERROR_THRESHOLD = 1e-4
CONVERGENCE_THRESHOLD = 1e-6


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(input_dim, hidden_layers):

    layers = []

    current_input_size = input_dim

    # --------------------------------------------------------
    # Hidden layers
    # Tanh is used for nonlinear regression.
    # The output layer remains Linear.
    # --------------------------------------------------------

    for hidden_size in hidden_layers:

        layers.append(
            Dense(
                current_input_size,
                hidden_size
            )
        )

        layers.append(
            Tanh()
        )

        current_input_size = hidden_size

    # --------------------------------------------------------
    # Output layer
    # Regression -> one output neuron
    # --------------------------------------------------------

    layers.append(
        Dense(
            current_input_size,
            1
        )
    )

    layers.append(
        Linear()
    )

    return FCNN(layers)


# ============================================================
# GET ARCHITECTURES
# ============================================================

def get_architectures(input_dim):

    # --------------------------------------------------------
    # Dataset 1: Univariate
    # One hidden layer
    # --------------------------------------------------------

    if input_dim == 1:

        return {
            "1-3-1": [3],
            "1-5-1": [5],
            "1-8-1": [8],
            "1-10-1": [10]
        }

    # --------------------------------------------------------
    # Dataset 2: Bivariate
    # One or two hidden layers
    # --------------------------------------------------------

    elif input_dim == 2:

        return {
            "2-3-1": [3],
            "2-5-1": [5],
            "2-8-1": [8],
            "2-5-5-1": [5, 5],
            "2-8-5-1": [8, 5]
        }

    else:

        raise ValueError(
            "Only univariate and bivariate "
            "regression are supported."
        )


# ============================================================
# TRAIN ONE ARCHITECTURE
# ============================================================

def train_architecture(
    input_dim,
    hidden_layers,
    X_train,
    Y_train,
    X_val,
    Y_val
):

    model = build_model(
        input_dim,
        hidden_layers
    )

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
# PREDICTION
# ============================================================

def predict(model, X):

    prediction = model.predict(X)

    return np.asarray(
        prediction
    ).reshape(-1)


# ============================================================
# EVALUATE
# ============================================================

def evaluate(model, X, Y):

    prediction = predict(
        model,
        X
    )

    metrics = Metrics(
        np.asarray(Y).reshape(-1),
        prediction
    )

    return {
        "prediction": prediction,
        "rmse": metrics.root_mean_squared_error(),
        "percent_rmse":
            metrics.percentage_root_mean_squared_error()
    }


# ============================================================
# PRINT ARCHITECTURE RESULT
# ============================================================

def print_architecture_result(
    name,
    train_result,
    validation_result,
    history
):

    print("\n" + "=" * 60)
    print(
        f"Architecture: {name}"
    )
    print("=" * 60)

    print(
        f"Train RMSE       : "
        f"{train_result['rmse']:.6f}"
    )

    print(
        f"Train %RMSE      : "
        f"{train_result['percent_rmse']:.6f}%"
    )

    print(
        f"Validation RMSE  : "
        f"{validation_result['rmse']:.6f}"
    )

    print(
        f"Validation %RMSE : "
        f"{validation_result['percent_rmse']:.6f}%"
    )

    print(
        f"Best epoch       : "
        f"{history['best_epoch']}"
    )

    print(
        f"Best validation error : "
        f"{history['best_validation_error']:.10f}"
    )



# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. CREATE RESULT DIRECTORY
    # ========================================================

    result_dir = "results/regression"

    os.makedirs(
        result_dir,
        exist_ok=True
    )

    # ========================================================
    # 2. LOAD DATA
    # ========================================================

    prepare = Data_preparation()

    data = prepare.load_data(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Last column = target
    # All previous columns = input
    # --------------------------------------------------------

    X = data[:, :-1]
    Y = data[:, -1]

    X = np.asarray(X)
    Y = np.asarray(Y)

    # --------------------------------------------------------
    # Ensure X is always 2-D
    # --------------------------------------------------------

    if X.ndim == 1:

        X = X.reshape(
            -1,
            1
        )

    input_dim = X.shape[1]

    # ========================================================
    # 3. SELECT ARCHITECTURES
    # ========================================================

    architectures = get_architectures(
        input_dim
    )

    # ========================================================
    # 4. TRAIN / VALIDATION / TEST SPLIT
    # ========================================================

    (
        X_train,
        X_val,
        X_test,
        Y_train,
        Y_val,
        Y_test
    ) = prepare.train_val_test_split(
        X,
        Y
    )

    print("\n")
    print("#" * 60)
    print("REGRESSION DATASET")
    print("#" * 60)

    print(
        "Dataset:",
        DATA_PATH
    )

    print(
        "Input dimensions:",
        input_dim
    )

    print(
        "Total samples:",
        len(X)
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_val)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    # ========================================================
    # 5. ARCHITECTURE EXPERIMENTS
    # ========================================================

    results = []

    for name, hidden_layers in architectures.items():

        model, history = train_architecture(
            input_dim,
            hidden_layers,
            X_train,
            Y_train,
            X_val,
            Y_val
        )

        # ----------------------------------------------------
        # Evaluate training data
        # ----------------------------------------------------

        train_result = evaluate(
            model,
            X_train,
            Y_train
        )

        # ----------------------------------------------------
        # Evaluate validation data
        # ----------------------------------------------------

        validation_result = evaluate(
            model,
            X_val,
            Y_val
        )

        print_architecture_result(
            name,
            train_result,
            validation_result,
            history
        )

        results.append({
            "name": name,
            "hidden_layers": hidden_layers,
            "model": model,
            "history": history,
            "train": train_result,
            "validation": validation_result
        })

    # ========================================================
    # 6. SELECT BEST ARCHITECTURE
    # ========================================================

    best_result = min(
        results,
        key=lambda result:
        result["validation"]["rmse"]
    )

    best_model = best_result["model"]
    best_history = best_result["history"]

    # ========================================================
    # 7. BEST ARCHITECTURE
    # ========================================================

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
        best_history[
            "best_validation_error"
        ]
    )

    print(
        "Validation RMSE:",
        best_result[
            "validation"
        ]["rmse"]
    )

    print(
        "Validation %RMSE:",
        best_result[
            "validation"
        ]["percent_rmse"]
    )

    # ========================================================
    # 8. FINAL TEST EVALUATION
    # ========================================================

    test_result = evaluate(
        best_model,
        X_test,
        Y_test
    )

    print("\n")
    print("#" * 60)
    print("TEST RESULTS - BEST ARCHITECTURE")
    print("#" * 60)

    print(
        "Test RMSE:",
        test_result["rmse"]
    )

    print(
        "Test %RMSE:",
        test_result["percent_rmse"]
    )

    # ========================================================
    # 9. VISUALIZATION
    # ========================================================

    plots = Plots()

    # --------------------------------------------------------
    # Error history
    #
    # Use existing line_plot because your current Plots
    # class does not contain plot_error_history().
    # --------------------------------------------------------

    epochs = np.arange(
        1,
        len(best_history["training_error"]) + 1
    )

    plots.line_plot(
        epochs,
        best_history["training_error"],
        title="Training Error vs Epoch",
        xlabel="Epoch",
        ylabel="Average Error",
        save_path=os.path.join(
            result_dir,
            "best_training_error_vs_epoch_bi.png"
        )
    )

    # --------------------------------------------------------
    # Validation error
    # --------------------------------------------------------

    if len(
        best_history["validation_error"]
    ) > 0:

        validation_epochs = np.arange(
            1,
            len(
                best_history[
                    "validation_error"
                ]
            ) + 1
        )

        plots.line_plot(
            validation_epochs,
            best_history[
                "validation_error"
            ],
            title="Validation Error vs Epoch",
            xlabel="Epoch",
            ylabel="Average Error",
            save_path=os.path.join(
                result_dir,
                "best_validation_error_vs_epoch_bi.png"
            )
        )

    # ========================================================
    # 10. TARGET VS MODEL OUTPUT
    # ========================================================

    if input_dim == 1:

        # ----------------------------------------------------
        # Univariate
        # ----------------------------------------------------

        train_result = best_result["train"]
        validation_result = best_result["validation"]

        plots.target_vs_model_plot(
            X_train[:, 0],
            Y_train,
            train_result["prediction"],
            title="Training: Target vs Model Output",
            xlabel="X",
            ylabel="Y",
            save_path=os.path.join(
                result_dir,
                "training_target_vs_model_bi.png"
            )
        )

        plots.target_vs_model_plot(
            X_val[:, 0],
            Y_val,
            validation_result["prediction"],
            title="Validation: Target vs Model Output",
            xlabel="X",
            ylabel="Y",
            save_path=os.path.join(
                result_dir,
                "validation_target_vs_model_bi.png"
            )
        )

        plots.target_vs_model_plot(
            X_test[:, 0],
            Y_test,
            test_result["prediction"],
            title="Test: Target vs Model Output",
            xlabel="X",
            ylabel="Y",
            save_path=os.path.join(
                result_dir,
                "test_target_vs_model_bi.png"
            )
        )

    else:

        # ----------------------------------------------------
        # Bivariate
        #
        # Use 3-D scatter:
        # X1, X2, model output
        # ----------------------------------------------------

        train_result = best_result["train"]
        validation_result = best_result["validation"]

        train_data_3d = np.column_stack(
            (
                X_train[:, 0],
                X_train[:, 1],
                train_result["prediction"]
            )
        )

        validation_data_3d = np.column_stack(
            (
                X_val[:, 0],
                X_val[:, 1],
                validation_result["prediction"]
            )
        )

        test_data_3d = np.column_stack(
            (
                X_test[:, 0],
                X_test[:, 1],
                test_result["prediction"]
            )
        )

        plots.regression_surface_plot(
            X_train,
            Y_train,
            best_model,
            title="Training: Target Output and Model Output",
            xlabel="X1",
            ylabel="X2",
            zlabel="Output",
            save_path=os.path.join(
                result_dir,
                "training_regression_surface_bi.png"
            )
        )

        plots.regression_surface_plot(
            X_val,
            Y_val,
            best_model,
            title="Validation: Target Output and Model Output",
            xlabel="X1",
            ylabel="X2",
            zlabel="Output",
            save_path=os.path.join(
                result_dir,
                "validation_regression_surface_bi.png"
            )
        )

        plots.regression_surface_plot(
            X_test,
            Y_test,
            best_model,
            title="Test: Target Output and Model Output",
            xlabel="X1",
            ylabel="X2",
            zlabel="Output",
            save_path=os.path.join(
                result_dir,
                "test_regression_surface_bi.png"
            )
        )

    # ========================================================
    # 11. TARGET VS MODEL OUTPUT SCATTER
    # ========================================================

    train_result = best_result["train"]
    validation_result = best_result["validation"]

    plots.scatter_plot(
        Y_train,
        train_result["prediction"],
        title="Training: Target vs Model Output",
        xlabel="Target Output",
        ylabel="Model Output",
        save_path=os.path.join(
            result_dir,
            "training_target_vs_model_scatter_bi.png"
        )
    )

    plots.scatter_plot(
        Y_val,
        validation_result["prediction"],
        title="Validation: Target vs Model Output",
        xlabel="Target Output",
        ylabel="Model Output",
        save_path=os.path.join(
            result_dir,
            "validation_target_vs_model_scatter_bi.png"
        )
    )

    plots.scatter_plot(
        Y_test,
        test_result["prediction"],
        title="Test: Target vs Model Output",
        xlabel="Target Output",
        ylabel="Model Output",
        save_path=os.path.join(
            result_dir,
            "test_target_vs_model_scatter_bi.png"
        )
    )

    # ========================================================
    # 12. NODE-OUTPUT VISUALIZATION
    # ========================================================
    # Generate plots for every hidden node and the output node
    # of the best architecture on train/validation/test data.
    plots.generate_regression_node_output_plots(
        best_model,
        X_train,
        X_val,
        X_test,
        result_dir,
        "univariate" if input_dim == 1 else "bivariate"
    )

    # ========================================================
    # 13. FINAL SUMMARY
    # ========================================================

    print("\n")
    print("#" * 60)
    print("FINAL SUMMARY")
    print("#" * 60)

    print(
        "Best architecture:",
        best_result["name"]
    )

    print(
        f"Train RMSE       : "
        f"{train_result['rmse']:.6f}"
    )

    print(
        f"Validation RMSE  : "
        f"{validation_result['rmse']:.6f}"
    )

    print(
        f"Test RMSE        : "
        f"{test_result['rmse']:.6f}"
    )

    print(
        f"Train %RMSE      : "
        f"{train_result['percent_rmse']:.6f}%"
    )

    print(
        f"Validation %RMSE : "
        f"{validation_result['percent_rmse']:.6f}%"
    )

    print(
        f"Test %RMSE       : "
        f"{test_result['percent_rmse']:.6f}%"
    )

    print(
        "\nResults saved to:",
        result_dir
    )


if __name__ == "__main__":
    main()