import numpy as np
import matplotlib.pyplot as plt

class Plots:

    def scatter_plot(self, X, Y, 
                     title='Data Visualization', xlabel='X-axis', 
                     ylabel='Y-axis', save_path='scatter_plot_univariate.png'):
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, marker='o', linestyle='-', color='blue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(save_path)  # Save the plot as a PNG file
        plt.show()

    def line_plot(self, X, Y, title='Data Visualization',
                  xlabel='X-axis', ylabel='Y-axis', 
                  save_path='line_plot_univariate.png'):
        
        plt.figure(figsize=(10, 6))
        plt.plot(X, Y, marker='o', linestyle='-', color='blue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(save_path)  # Save the plot as a PNG file
        plt.show()
    
    def scatter_line_plot(self, X, Y, 
                          title='Data Visualization', xlabel='X-axis', 
                          ylabel='Y-axis', 
                          save_path='scatter_line_plot_univariate.png'):
        
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, marker='o', color='green', label='Data Points')
        plt.plot(X, Y, linestyle='-', color='red', label='Line Plot')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)  # Save the plot as a PNG file
        plt.show()

    def target_vs_model_plot(self, X, y_true, y_pred, 
                             title='Target vs Model Output', xlabel='X', 
                             ylabel='Y', save_path='target_vs_model.png'):

        order = np.argsort(X)

        X_sorted = X[order]
        y_true_sorted = y_true[order]
        y_pred_sorted = y_pred[order]

        plt.figure(figsize=(10, 6))

        plt.scatter(X_sorted, y_true_sorted, marker='o', label='Target Output')

        plt.plot(X_sorted, y_pred_sorted, label='Model Output')

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()

        plt.savefig(save_path)
        plt.show()
        plt.close()
        
    def plot_3d_scatter(self, data, title='3D Scatter Plot',
                        xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis',
                        save_path='3D_scatter_plot.png'):
        
        # 2. Initialize a 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection='3d')

        # 3. Define styling properties
        colors = ['red', 'green', 'blue']
        class_names = ['Class 1', 'Class 2', 'Class 3']

        # 4. Loop through each class and plot using boolean masking
        for class_idx in [0, 1, 2]:
            # Filter rows belonging to the current class
            class_mask = data[:, -1] == class_idx
            class_data = data[class_mask]
            
            # Extract X, Y, Z coordinates
            x = class_data[:, 0]
            y = class_data[:, 1]
            z = class_data[:, 2]
            
            # Plot the isolated data points
            ax.scatter(x, y, z, c=colors[class_idx], label=class_names[class_idx], s=40)

        # 5. Format labels and display legend
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('3D Scatter Plot for 3 Classes')
        ax.legend()

        plt.savefig(save_path)  # Save the plot as a PNG file
        plt.show()
        
    def plot_cm_from_scratch(self,cm, labels, cmap='Blues', save_path="confusion_matrix.png"):
        """
        Plots a 2D confusion matrix array graphically using matplotlib.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Draw the matrix grid cells with background color scaling
        cax = ax.matshow(cm, cmap=cmap)
        fig.colorbar(cax) # Add color scale bar
        
        num_classes = len(labels)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        
        # Label the axes with your custom strings
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # Move x-axis labels to the bottom to match scikit-learn standard
        ax.xaxis.set_ticks_position('bottom')
        
        # Add text annotations inside each box cell
        threshold = cm.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                # Change text color dynamically based on background darkness
                color = "white" if cm[i, j] > threshold else "black"
                ax.text(j, i, str(int(cm[i, j])), 
                        va='center', ha='center', color=color, fontsize=12)
                
        plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
        plt.ylabel('True Label', fontsize=12, labelpad=10)
        plt.title('Confusion Matrix', fontsize=14, pad=20)
        plt.savefig(save_path)
        plt.show()
        
    def plot_pair_wise(self, X_train, Y_train, classifiers, class1, 
                       class2, savepath="pairwise_plot.png"):
        masks = (Y_train == class1) | (Y_train == class2)
        
        X_filtered = X_train[masks]
        Y_filtered = Y_train[masks]
        
        model = classifiers[(class1,class2)]['model']
        
        x1_min = X_filtered[:, 0].min() - 1
        x1_max = X_filtered[:, 0].max() + 1

        x2_min = X_filtered[:, 1].min() - 1
        x2_max = X_filtered[:, 1].max() + 1

        x1_values = np.linspace(
            x1_min,
            x1_max,
            300
        )

        x2_values = np.linspace(
            x2_min,
            x2_max,
            300
        )

        xx, yy = np.meshgrid(
            x1_values,
            x2_values
        )

        grid = np.c_[
            xx.ravel(),
            yy.ravel()
        ]
        
        predictions = model.predict(grid)

        # Convert binary prediction back to original class
        # 0 → class1
        # 1 → class2

        predictions = np.where(
            predictions == 0,
            class1,
            class2
        )

        predictions = predictions.reshape(
            xx.shape
        )
        
        plt.figure(figsize=(10, 7))

        plt.contourf(
            xx,
            yy,
            predictions,
            alpha=0.25
        )

        for class_label in [class1, class2]:

            points = X_filtered[
                Y_filtered == class_label
            ]

            plt.scatter(
                points[:, 0],
                points[:, 1],
                label=f"Class {class_label + 1}"
            )   
        
        weights = model.weights

        if abs(weights[2]) > 1e-12:

            boundary_x = np.linspace(
                x1_min,
                x1_max,
                300
            )

            boundary_y = (
                -(weights[0] + weights[1] * boundary_x)
                / weights[2]
            )

            plt.plot(
                boundary_x,
                boundary_y,
                label="Decision Boundary"
            )

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------

        plt.xlabel("X1")
        plt.ylabel("X2")

        plt.title(
            f"Decision Region: "
            f"Class {class1 + 1} vs "
            f"Class {class2 + 1}"
        )

        plt.legend()
        plt.grid(True)

        plt.savefig(
            savepath,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

    def plot_combined_ovo_decision_region(
        self,
        X_train,
        Y_train,
        classifier,
        save_path="combined_ovo_decision_region.png"
    ):

        # ---------------------------------------------------------
        # Determine plotting range from training data
        # ---------------------------------------------------------

        x1_min = X_train[:, 0].min() - 1
        x1_max = X_train[:, 0].max() + 1

        x2_min = X_train[:, 1].min() - 1
        x2_max = X_train[:, 1].max() + 1

        # ---------------------------------------------------------
        # Create 2-D grid
        # ---------------------------------------------------------

        x1_values = np.linspace(x1_min, x1_max, 500)
        x2_values = np.linspace(x2_min, x2_max, 500)

        xx, yy = np.meshgrid(x1_values, x2_values)

        grid = np.c_[
            xx.ravel(),
            yy.ravel()
        ]

        # ---------------------------------------------------------
        # Get FINAL OVO prediction for every grid point
        # ---------------------------------------------------------

        predictions = classifier.predict(grid)

        predictions = predictions.reshape(xx.shape)

        # ---------------------------------------------------------
        # Plot final decision regions
        # ---------------------------------------------------------

        plt.figure(figsize=(10, 7))

        from matplotlib.colors import ListedColormap

        region_colors = ListedColormap([
            '#ADD8E6',   # Class 1
            '#FFD580',   # Class 2
            '#90EE90'    # Class 3
        ])

        plt.contourf(
            xx,
            yy,
            predictions,
            levels=[-0.5, 0.5, 1.5, 2.5],
            cmap=region_colors,
            alpha=0.5
        )

        # ---------------------------------------------------------
        # Plot training data
        # ---------------------------------------------------------

        class_colors = [
            'blue',
            'orange',
            'green'
        ]

        for class_label in range(3):

            points = X_train[
                Y_train == class_label
            ]

            plt.scatter(
                points[:, 0],
                points[:, 1],
                color=class_colors[class_label],
                label=f"Class {class_label + 1}",
                s=25
            )

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------

        plt.xlabel("X1")
        plt.ylabel("X2")

        plt.title(
            "Multi-class Decision Region"
        )

        plt.legend()

        plt.grid(True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()
    
    def plot_error_history(
        self,
        history,
        save_path="error_vs_epoch.png"
    ):

        epochs = history["epochs"]
        train_error = history["train_error"]
        validation_error = history["validation_error"]

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            train_error,
            label="Training Error"
        )

        plt.plot(
            epochs,
            validation_error,
            label="Validation Error"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Average Error")

        plt.title(
            "Average Error vs Epoch"
        )

        plt.grid(True)
        plt.legend()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()
    
    def plot_fcnn_decision_region(
        self,
        X_train,
        Y_train,
        model,
        save_path="decision_region.png"
    ):

        x1_min = X_train[:, 0].min() - 1
        x1_max = X_train[:, 0].max() + 1

        x2_min = X_train[:, 1].min() - 1
        x2_max = X_train[:, 1].max() + 1

        x1_values = np.linspace(
            x1_min,
            x1_max,
            300
        )

        x2_values = np.linspace(
            x2_min,
            x2_max,
            300
        )

        xx, yy = np.meshgrid(
            x1_values,
            x2_values
        )

        grid = np.c_[
            xx.ravel(),
            yy.ravel()
        ]

        output = model.predict(grid)

        predictions = np.argmax(
            output,
            axis=1
        )

        predictions = predictions.reshape(
            xx.shape
        )

        plt.figure(figsize=(10, 7))

        plt.contourf(
            xx,
            yy,
            predictions,
            alpha=0.25
        )

        classes = np.unique(Y_train)

        for class_label in classes:

            points = X_train[
                Y_train == class_label
            ]

            plt.scatter(
                points[:, 0],
                points[:, 1],
                label=f"Class {class_label + 1}",
                s=25
            )

        plt.xlabel("X1")
        plt.ylabel("X2")

        plt.title(
            "FCNN Decision Region"
        )

        plt.legend()
        plt.grid(True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()
    
    def plot_error_history(
        self,
        history,
        save_path="error_vs_epoch.png"
    ):

        training_error = history["training_error"]
        validation_error = history["validation_error"]

        epochs = np.arange(
            1,
            len(training_error) + 1
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            training_error,
            label="Training Error"
        )

        plt.plot(
            epochs,
            validation_error,
            label="Validation Error"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Average Error")

        plt.title(
            "Average Error vs Epoch"
        )

        plt.grid(True)
        plt.legend()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()
    
    def plot_fcnn_decision_region(
        self,
        X_train,
        Y_train,
        model,
        save_path="fcnn_decision_region.png"
    ):

        x1_min = X_train[:, 0].min() - 1
        x1_max = X_train[:, 0].max() + 1

        x2_min = X_train[:, 1].min() - 1
        x2_max = X_train[:, 1].max() + 1

        x1_values = np.linspace(
            x1_min,
            x1_max,
            300
        )

        x2_values = np.linspace(
            x2_min,
            x2_max,
            300
        )

        xx, yy = np.meshgrid(
            x1_values,
            x2_values
        )

        grid = np.c_[
            xx.ravel(),
            yy.ravel()
        ]

        # FCNN output
        output = model.predict(grid)

        # Convert output vector to class
        predictions = np.argmax(
            output,
            axis=1
        )

        predictions = predictions.reshape(
            xx.shape
        )

        plt.figure(figsize=(10, 7))

        plt.contourf(
            xx,
            yy,
            predictions,
            alpha=0.25
        )

        for class_label in np.unique(Y_train):

            points = X_train[
                Y_train == class_label
            ]

            plt.scatter(
                points[:, 0],
                points[:, 1],
                label=f"Class {class_label + 1}",
                s=25
            )

        plt.xlabel("X1")
        plt.ylabel("X2")

        plt.title(
            "FCNN Decision Region"
        )

        plt.legend()
        plt.grid(True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()
    
    def plot_node_outputs_3d(
        self,
        X,
        Y,
        model,
        dataset_name,
        save_dir
    ):

        import os

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Get activations for every example
        # ----------------------------------------------------

        all_outputs = []

        for x in X:

            outputs = model.forward_with_activations(x)

            all_outputs.append(outputs)

        # ----------------------------------------------------
        # Convert:
        #
        # example -> layer -> node
        #
        # into:
        #
        # layer -> examples -> nodes
        # ----------------------------------------------------

        number_of_layers = len(
            all_outputs[0]
        )

        for layer_index in range(
            number_of_layers
        ):

            layer_outputs = np.array([
                example_outputs[layer_index]
                for example_outputs in all_outputs
            ])

            number_of_nodes = (
                layer_outputs.shape[1]
            )

            for node_index in range(
                number_of_nodes
            ):

                z = layer_outputs[
                    :, node_index
                ]

                fig = plt.figure(
                    figsize=(9, 7)
                )

                ax = fig.add_subplot(
                    projection="3d"
                )

                for class_label in np.unique(Y):

                    mask = (
                        Y == class_label
                    )

                    ax.scatter(
                        X[mask, 0],
                        X[mask, 1],
                        z[mask],
                        label=f"Class {class_label + 1}",
                        s=15
                    )

                if layer_index == number_of_layers - 1:

                    layer_name = "Output"

                else:

                    layer_name = (
                        f"Hidden Layer "
                        f"{layer_index + 1}"
                    )

                ax.set_xlabel("X1")
                ax.set_ylabel("X2")
                ax.set_zlabel(
                    f"Node {node_index + 1} Output"
                )

                ax.set_title(
                    f"{dataset_name} - "
                    f"{layer_name} - "
                    f"Node {node_index + 1}"
                )

                ax.legend()

                filename = (
                    f"{dataset_name}_"
                    f"{layer_name.replace(' ', '_')}_"
                    f"Node_{node_index + 1}.png"
                )

                plt.savefig(
                    os.path.join(
                        save_dir,
                        filename
                    ),
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.show()
                plt.close()
    def regression_surface_plot(
        self,
        X,
        Y,
        model,
        title='Regression Surface',
        xlabel='X1',
        ylabel='X2',
        zlabel='Y',
        save_path='regression_surface.png'
    ):
        """
        Plot bivariate regression data and the model's
        predicted regression surface.

        X must have shape (N, 2).
        Y must contain the actual target values.
        model must provide a predict(X) method.
        """

        X = np.asarray(X)
        Y = np.asarray(Y).reshape(-1)

        if X.shape[1] != 2:
            raise ValueError(
                "regression_surface_plot requires exactly "
                "2 input features."
            )

        # ---------------------------------------------------------
        # Determine plotting range
        # ---------------------------------------------------------

        x1_min = X[:, 0].min()
        x1_max = X[:, 0].max()

        x2_min = X[:, 1].min()
        x2_max = X[:, 1].max()

        # Small padding around the data
        x1_range = x1_max - x1_min
        x2_range = x2_max - x2_min

        x1_min -= 0.05 * x1_range
        x1_max += 0.05 * x1_range

        x2_min -= 0.05 * x2_range
        x2_max += 0.05 * x2_range

        # ---------------------------------------------------------
        # Create grid
        # ---------------------------------------------------------

        x1_values = np.linspace(x1_min, x1_max, 50)
        x2_values = np.linspace(x2_min, x2_max, 50)

        xx, yy = np.meshgrid(
            x1_values,
            x2_values
        )

        grid = np.column_stack(
            (
                xx.ravel(),
                yy.ravel()
            )
        )

        # ---------------------------------------------------------
        # Model prediction over the complete grid
        # ---------------------------------------------------------

        predictions = model.predict(grid)

        predictions = np.asarray(
            predictions
        ).reshape(-1)

        zz = predictions.reshape(
            xx.shape
        )

        # ---------------------------------------------------------
        # Create 3D plot
        # ---------------------------------------------------------

        fig = plt.figure(figsize=(10, 7))

        ax = fig.add_subplot(
            projection='3d'
        )

        # Actual target points
        ax.scatter(
            X[:, 0],
            X[:, 1],
            Y,
            label='Target Output',
            s=20
        )

        # Predicted regression surface
        ax.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.5
        )

        # ---------------------------------------------------------
        # Labels
        # ---------------------------------------------------------

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        ax.set_title(title)

        ax.legend()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
        plt.close()
        
    def target_vs_prediction_plot(
        self,
        y_true,
        y_pred,
        title='Target vs Model Output',
        save_path='target_vs_prediction.png'
    ):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        minimum = min(
            y_true.min(),
            y_pred.min()
        )

        maximum = max(
            y_true.max(),
            y_pred.max()
        )

        plt.figure(figsize=(10, 6))

        plt.scatter(
            y_true,
            y_pred,
            label='Predictions'
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle='--',
            label='Ideal'
        )

        plt.xlabel("Target Output")
        plt.ylabel("Model Output")

        plt.title(title)

        plt.grid(True)
        plt.legend()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )

        plt.show()
        plt.close()
    
    def generate_regression_node_output_plots(
        self,
        model,
        X_train,
        X_val,
        X_test,
        result_dir,
        dataset_type
    ):
        """
        Generate node-output plots for the selected best
        regression architecture on training, validation and
        test data.
        """
        import os

        base_dir = os.path.join(
            result_dir,
            "node_outputs",
            dataset_type
        )

        self.plot_regression_node_outputs(
            X_train,
            model,
            "Training",
            os.path.join(base_dir, "Training")
        )

        self.plot_regression_node_outputs(
            X_val,
            model,
            "Validation",
            os.path.join(base_dir, "Validation")
        )

        self.plot_regression_node_outputs(
            X_test,
            model,
            "Test",
            os.path.join(base_dir, "Test")
        )
    def plot_regression_node_outputs(
        self,
        X,
        model,
        dataset_name,
        save_dir
    ):
        """
        Plot the output of every hidden-layer node and the
        regression output node for the trained FCNN.

        The FCNN implementation stores the output of each
        activation layer in ``activation_value``. Therefore,
        this method does not depend on a separate
        forward_with_activations() method.

        Univariate regression:
            X versus node output (2-D plot).

        Bivariate regression:
            X1, X2 versus node output (3-D plot).
        """
        import os

        X = np.asarray(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2 or X.shape[1] not in (1, 2):
            raise ValueError(
                "Regression node-output visualization requires "
                "one or two input features."
            )

        os.makedirs(save_dir, exist_ok=True)

        # ---------------------------------------------------------
        # Identify activation layers from the actual FCNN model.
        # Dense layers are skipped because the assignment asks
        # for node outputs after the activation functions.
        # ---------------------------------------------------------
        activation_layers = []

        for layer in model.layers:
            class_name = layer.__class__.__name__

            if class_name == "Tanh":
                activation_layers.append(
                    ("Hidden Layer", layer)
                )

            elif class_name == "Linear":
                activation_layers.append(
                    ("Output", layer)
                )

        if not activation_layers:
            raise ValueError(
                "No Tanh or Linear activation layers were found "
                "in the supplied FCNN model."
            )

        # ---------------------------------------------------------
        # Run each sample through the trained network and record
        # the actual activation_value of every activation layer.
        # ---------------------------------------------------------
        outputs_by_layer = [
            [] for _ in activation_layers
        ]

        for x in X:
            model.forward(x)

            for layer_index, (_, layer) in enumerate(
                activation_layers
            ):
                if layer.activation_value is None:
                    raise ValueError(
                        "Activation value is unavailable after "
                        "model.forward()."
                    )

                output = np.asarray(
                    layer.activation_value
                ).reshape(-1)

                outputs_by_layer[layer_index].append(
                    output.copy()
                )

        outputs_by_layer = [
            np.asarray(layer_outputs)
            for layer_outputs in outputs_by_layer
        ]

        # ---------------------------------------------------------
        # Plot every node separately.
        # ---------------------------------------------------------
        for layer_index, (
            layer_name,
            _
        ) in enumerate(activation_layers):

            layer_outputs = outputs_by_layer[layer_index]
            number_of_nodes = layer_outputs.shape[1]

            for node_index in range(number_of_nodes):
                node_output = layer_outputs[:, node_index]

                if X.shape[1] == 1:
                    # Sort only for a visually meaningful response
                    # curve against the single input variable.
                    order = np.argsort(X[:, 0])
                    x_sorted = X[order, 0]
                    output_sorted = node_output[order]

                    plt.figure(figsize=(10, 6))
                    plt.scatter(
                        x_sorted,
                        output_sorted,
                        s=18,
                        label="Node Output"
                    )
                    plt.plot(
                        x_sorted,
                        output_sorted,
                        linewidth=1,
                        label="Node Response"
                    )

                    plt.xlabel("Input X")
                    plt.ylabel(
                        f"Node {node_index + 1} Output"
                    )
                    plt.title(
                        f"{dataset_name}: {layer_name} - "
                        f"Node {node_index + 1}"
                    )
                    plt.grid(True)
                    plt.legend()
                    plt.tight_layout()

                else:
                    fig = plt.figure(figsize=(9, 7))
                    ax = fig.add_subplot(
                        projection="3d"
                    )

                    ax.scatter(
                        X[:, 0],
                        X[:, 1],
                        node_output,
                        s=12
                    )

                    ax.set_xlabel("X1")
                    ax.set_ylabel("X2")
                    ax.set_zlabel(
                        f"Node {node_index + 1} Output"
                    )
                    ax.set_title(
                        f"{dataset_name}: {layer_name} - "
                        f"Node {node_index + 1}"
                    )
                    plt.tight_layout()

                filename = (
                    f"{dataset_name}_"
                    f"{layer_name.replace(' ', '_')}_"
                    f"Node_{node_index + 1}.png"
                )

                plt.savefig(
                    os.path.join(save_dir, filename),
                    dpi=300,
                    bbox_inches="tight"
                )
                plt.close()