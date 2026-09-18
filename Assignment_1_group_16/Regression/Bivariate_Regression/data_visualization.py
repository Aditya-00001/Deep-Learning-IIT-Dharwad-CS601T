import numpy as np
import matplotlib.pyplot as plt

class Plot:

    def scatter_plot(self, X, Y, title='Data Visualization', xlabel='X-axis', ylabel='Y-axis', save_path='scatter_plot_univariate.png'):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, marker='o', linestyle='-', color='blue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(save_path)  # Save the plot as a PNG file

    def line_plot(self, X, Y, title='Data Visualization', xlabel='X-axis', ylabel='Y-axis', save_path='line_plot_univariate.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(X, Y, marker='o', linestyle='-', color='blue')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(save_path)  # Save the plot as a PNG file
    
    def scatter_line_plot(self, X, Y, title='Data Visualization', xlabel='X-axis', ylabel='Y-axis', save_path='scatter_line_plot_univariate.png'):
        plt.figure(figsize=(10, 6))
        plt.scatter(X, Y, marker='o', color='green', label='Data Points')
        plt.plot(X, Y, linestyle='-', color='red', label='Line Plot')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)  # Save the plot as a PNG file

    def target_vs_model_plot(self, X, y_true, y_pred, title='Target vs Model Output', xlabel='X', ylabel='Y', save_path='target_vs_model.png'):

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
    
    def regression_surface_plot(self, X, y_true, weights, title='Target Data and Linear Regression Plane', save_path='regression_surface.png'):

        from mpl_toolkits.mplot3d import Axes3D

        x1 = X[:, 0]
        x2 = X[:, 1]

        x1_grid = np.linspace(x1.min(), x1.max(), 30)
        x2_grid = np.linspace(x2.min(), x2.max(), 30)

        X1, X2 = np.meshgrid(x1_grid, x2_grid)

        W0 = weights[0]
        W1 = weights[1]
        W2 = weights[2]

        Y_pred = W0 + W1 * X1 + W2 * X2

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Actual target points
        ax.scatter(
            x1,
            x2,
            y_true,
            label='Target Output'
        )

        # Regression plane
        ax.plot_surface(
            X1,
            X2,
            Y_pred,
            alpha=0.5
        )

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')

        ax.set_title(title)
        ax.legend()

        plt.savefig(save_path)
        plt.show()
        plt.close()