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