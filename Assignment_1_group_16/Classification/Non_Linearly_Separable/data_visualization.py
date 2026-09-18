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