import numpy as np

np.random.seed(42)  # For reproducibility
class Trainer:

    def __init__(
        self,
        model,
        loss_function,
        learning_rate=0.01,
        max_epochs=1000,
        error_threshold=None,
        convergence_threshold=None,
        validation_early_stopping=False
    ):

        self.model = model
        self.loss_function = loss_function

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.error_threshold = error_threshold
        self.convergence_threshold = convergence_threshold

        self.validation_early_stopping = (
            validation_early_stopping
        )

        self.error_history = []
        self.validation_error_history = []

        self.best_validation_error = np.inf
        self.best_weights = None
        self.best_epoch = None

    # -----------------------------------------
    # SGD weight update
    # -----------------------------------------

    def _update_weights(self):

        for layer in self.model.layers:

            if hasattr(layer, "dW"):

                layer.weights -= self.learning_rate * layer.dW


    # -----------------------------------------
    # Save model weights
    # -----------------------------------------

    def _save_weights(self):

        return [
            layer.weights.copy()
            for layer in self.model.layers
            if hasattr(layer, "weights")
        ]

    # -----------------------------------------
    # Restore model weights
    # -----------------------------------------

    def _restore_weights(self, weights):

        index = 0

        for layer in self.model.layers:

            if hasattr(layer, "weights"):

                layer.weights = weights[index].copy()

                index += 1

    # -----------------------------------------
    # Compute average error
    # -----------------------------------------

    def _compute_error(self, X, y):

        total_error = 0.0

        for x, target in zip(X, y):

            prediction = self.model.forward(x)

            error = self.loss_function.forward(
                target,
                prediction
            )

            total_error += error

        return total_error / len(X)

    # -----------------------------------------
    # Training
    # -----------------------------------------

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None
    ):

        previous_error = None

        for epoch in range(self.max_epochs):

            # =================================
            # Shuffle training examples
            # =================================

            indices = np.random.permutation(
                len(X_train)
            )

            total_error = 0.0

            # =================================
            # Pattern-mode SGD
            # =================================

            for index in indices:

                x = X_train[index]
                target = y_train[index]

                # Forward propagation
                prediction = self.model.forward(x)

                # Instantaneous squared error
                error = self.loss_function.forward(
                    target,
                    prediction
                )

                # Gradient of loss
                gradient = self.loss_function.backward()

                # Backpropagation
                self.model.backward(gradient)

                # Update weights
                self._update_weights()

                total_error += error

            # =================================
            # Average training error
            # =================================

            average_error = (
                total_error / len(X_train)
            )

            self.error_history.append(
                average_error
            )

            # =================================
            # Validation
            # =================================

            if X_val is not None and y_val is not None:

                validation_error = self._compute_error(
                    X_val,
                    y_val
                )

                self.validation_error_history.append(
                    validation_error
                )

                # -----------------------------
                # Save best model
                # -----------------------------

                if validation_error < self.best_validation_error:

                    self.best_validation_error = (
                        validation_error
                    )

                    self.best_weights = (
                        self._save_weights()
                    )

                    self.best_epoch = epoch + 1

                # -----------------------------
                # Validation early stopping
                # -----------------------------

                elif self.validation_early_stopping:

                    print(
                        f"Validation error increased "
                        f"at epoch {epoch + 1}"
                    )

                    break

            # =================================
            # Error threshold
            # =================================

            if (
                self.error_threshold is not None
                and average_error <= self.error_threshold
            ):

                print(
                    f"Error threshold reached "
                    f"at epoch {epoch + 1}"
                )

                break

            # =================================
            # Successive error difference
            # =================================

            if previous_error is not None:

                difference = abs(
                    average_error - previous_error
                )

                if (
                    self.convergence_threshold
                    is not None
                    and difference
                    <= self.convergence_threshold
                ):

                    print(
                        f"Convergence reached "
                        f"at epoch {epoch + 1}"
                    )

                    break

            previous_error = average_error

        # =====================================
        # Restore best validation model
        # =====================================

        if self.best_weights is not None:

            self._restore_weights(
                self.best_weights
            )

        return {
            "training_error":
                self.error_history,

            "validation_error":
                self.validation_error_history,

            "best_epoch":
                self.best_epoch,

            "best_validation_error":
                self.best_validation_error
        }