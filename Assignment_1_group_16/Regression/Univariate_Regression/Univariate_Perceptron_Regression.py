from Preprocessing import train_test_split
from data_visualization import Plot
import numpy as np
from metrics import Metrics

class Perceptron_Regression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
    def fit(self, X, Y):
        train_error=[]
        
        prev_error = float('inf')
        
        X=np.hstack((np.ones((X.shape[0], 1)), X)) # for bias term
        
        n_samples, n_features = X.shape
        np.random.seed(42)
        self.weights = np.random.randn(n_features)*0.1
        for epoch in range(self.epochs):
            errors = []
            
            for n in range(n_samples):
                
                x = X[n]
                y = Y[n]
                sn  = np.dot(x, self.weights) 
                error = 0.5 * (y - sn) ** 2
                errors.append(error)
                self.weights -= self.lr * (sn - y) * x
            
            avg_error = np.mean(errors)
            train_error.append(avg_error)
            
            if epoch>=50 and abs(prev_error - avg_error) < 1e-8:
                print(abs(prev_error - avg_error))
                break
            prev_error = avg_error
        return train_error

    def predict(self, X):
        X=np.hstack((np.ones((X.shape[0], 1)), X))
        return np.dot(X, self.weights)
    
    def get_weights(self):
        return self.weights
    

data = np.loadtxt('16.csv', delimiter=',')
X_train, X_test, Y_train, Y_test = train_test_split(data[:, 0], data[:, 1], test_size=0.3)


PR = Perceptron_Regression(lr=0.01, epochs=1000)
# Training
history = PR.fit(
    X_train.reshape(-1, 1),
    Y_train
)

# Predictions of training data
yhat_train = PR.predict(
    X_train.reshape(-1, 1)
)

# Predictions of testing data
yhat_test = PR.predict(
    X_test.reshape(-1, 1)
)

# Final weights after training
print("\nFinal weights:")
print(PR.get_weights())


train_metrics = Metrics(Y_train, yhat_train)
test_metrics = Metrics(Y_test, yhat_test)

print("TRAINING RESULTS ")
print("RMSE:", train_metrics.root_mean_squared_error())
print("% RMSE:", train_metrics.percentage_root_mean_squared_error())

print("TESTING RESULTS")
print("RMSE:", test_metrics.root_mean_squared_error())
print("% RMSE:", test_metrics.percentage_root_mean_squared_error())






plot = Plot()

# 1. Average Error vs Epochs
plot.line_plot(
    range(1, len(history) + 1),
    history,
    title='Average Error vs Epochs',
    xlabel='Epochs',
    ylabel='Average Error',
    save_path='average_error_vs_epochs.png'
)

# 2. Target vs Model - Training
plot.target_vs_model_plot(
    X_train,
    Y_train,
    yhat_train,
    title='Training Data: Target Output vs Model Output',
    xlabel='X',
    ylabel='Y',
    save_path='training_target_vs_model.png'
)

# 3. Target vs Model - Testing
plot.target_vs_model_plot(
    X_test,
    Y_test,
    yhat_test,
    title='Testing Data: Target Output vs Model Output',
    xlabel='X',
    ylabel='Y',
    save_path='testing_target_vs_model.png'
)

# 4. Target output vs Model output - Training
plot.scatter_plot(
    Y_train,
    yhat_train,
    title='Training: Target Output vs Model Output',
    xlabel='Target Output',
    ylabel='Model Output',
    save_path='training_scatter.png'
)

# 5. Target output vs Model output - Testing
plot.scatter_plot(
    Y_test,
    yhat_test,
    title='Testing: Target Output vs Model Output',
    xlabel='Target Output',
    ylabel='Model Output',
    save_path='testing_scatter.png'
)
