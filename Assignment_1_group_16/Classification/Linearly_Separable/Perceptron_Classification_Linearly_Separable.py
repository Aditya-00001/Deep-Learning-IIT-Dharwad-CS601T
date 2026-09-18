import numpy as np
from Preprocessing import Data_preparation
from metrics import Metrics
from data_visualization import Plots
#Perceptron Binary Class
class Perceptron_Binary:
    def __init__(self, lr, epochs, activation="logistic"):
        self.weights = None
        self.epochs=epochs
        self.lr=lr
        if activation not in ["logistic", "tanh"]:
            raise ValueError(
                "Activation must be 'logistic' or 'tanh'"
            )

        self.activation = activation
    
    def logistic(self, x):
        return 1/(1+np.exp(-x))
    
    def logistic_derivative(self,x):
        value = self.logistic(x)
        return value*(1-value)
    
    def tanh(self,x):
        pos_exp = np.exp(x)
        neg_exp = np.exp(-x)
        return (pos_exp - neg_exp)/(pos_exp + neg_exp)
    
    def tanh_derivative(self, x):
        value = self.tanh(x)
        return 1 - value**2 
    
    def activation_function(self, x):
        if self.activation == "logistic":
            return self.logistic(x)
        else:
            return self.tanh(x)
    
    def activation_derivative(self, x):
        if self.activation == "logistic":
            return self.logistic_derivative(x)
        else:
            return self.tanh_derivative(x)
    
    def fit(self, X, Y):
        np.random.seed(42)  # Set a random seed for reproducibility
        train_error = []
        # Add a bias term to the input features
        X=np.hstack((np.ones((X.shape[0], 1)), X))
        
        n_samples, n_features = X.shape
        self.weights = np.random.uniform(-0.01, 0.01, n_features)
        
        for epoch in range(self.epochs):
            epoch_error = 0
            for i in range(n_samples):
                x, y = X[i], Y[i]
                if self.activation == "tanh":
                    target = 1 if y == 1 else -1
                else:
                    target = y
                linear_output = np.dot(x, self.weights)
                prediction = self.activation_function(linear_output)
                error = 0.5 * (target - prediction) ** 2
                epoch_error+=error
                dw = self.lr * (target - prediction) * self.activation_derivative(linear_output) * x
                self.weights += dw
            
            average_error = epoch_error / n_samples
            train_error.append(average_error)
            if (epoch>=50 and abs(train_error[-1] - train_error[-2]) < 1e-4):
                break
        return train_error
    
    def predict(self, X):
        # Add a bias term to the input features
        X=np.hstack((np.ones((X.shape[0], 1)), X))
        
        linear_output = np.dot(X, self.weights)
        
        output = self.activation_function(linear_output)

        if self.activation == "logistic":
            predictions = (output >= 0.5).astype(int)
        else:
            predictions = (output >= 0).astype(int)
        
        return predictions
    
    def decision_function(self, X):

        X = np.hstack(
            (np.ones((X.shape[0], 1)), X)
        )

        return np.dot(
            X,
            self.weights
        )
    
# Multiclass Perceptron using One-vs-One (OvO) strategy

class Perceptron_Multiclass_OVO:
    def __init__(self, lr, epochs, activation="logistic"):
        self.classifiers = {}
        self.epochs=epochs
        self.lr = lr
        self.activation=activation
    
    def fit(self, X, Y):
        data_class = [(0,1), (0, 2), (1, 2)]
        for class_1, class_2 in data_class:
            
            mask = (Y==class_1) | (Y==class_2)
            
            X_filtered = X[mask]
            Y_filtered = Y[mask]
            
            
            # Convert the labels to binary (0 for class_1, 1 for class_2)
            Y_binary = np.where(Y_filtered == class_1, 0, 1)
            model = Perceptron_Binary(lr=self.lr, epochs=self.epochs, activation=self.activation)
            train_error = model.fit(X_filtered, Y_binary)
            
            self.classifiers[(class_1, class_2)] = {
                'model': model,
                'train_error': train_error
            }
            
            print(
                f"Class {class_1 + 1} vs Class {class_2 + 1}"
            )
            print(
                f"Training samples: {len(X_filtered)}"
            )
            print(
                f"Epochs: {len(train_error)}"
            )
            print(
                f"Final error: {train_error[-1]}"
            )
            print()
            
        return self.classifiers 
    
    def predict(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, 3))  # Assuming 3 classes
        for pair in self.classifiers:
            model = self.classifiers[pair]['model']
            predictions = model.predict(X)
            for i,prediction in enumerate(predictions):
                if prediction == 0:
                    votes[i, pair[0]] += 1
                else:
                    votes[i, pair[1]] += 1
        final_predictions = np.argmax(votes, axis=1)
        return final_predictions
    
    def predict_pair(self, X, class_1, class_2):

        model = self.classifiers[
            (class_1, class_2)
        ]['model']

        binary_prediction = model.predict(X)

        prediction = np.where(
            binary_prediction == 0,
            class_1,
            class_2
        )

        return prediction
    
    def predict_votes(self, X):

        n_samples = X.shape[0]

        votes = np.zeros((n_samples, 3), dtype=int)

        pair_predictions = {}

        for pair in self.classifiers:

            model = self.classifiers[pair]['model']

            predictions = model.predict(X)

            pair_predictions[pair] = predictions

            for i, prediction in enumerate(predictions):

                if prediction == 0:
                    votes[i, pair[0]] += 1
                else:
                    votes[i, pair[1]] += 1

        final_predictions = np.argmax(votes, axis=1)

        return final_predictions, votes, pair_predictions


np.random.seed(42)  # Set a random seed for reproducibility   
preprocessing = Data_preparation()
Data_X, Data_y = preprocessing.prepare_data('Class1.txt', 'Class2.txt', 'Class3.txt')
classes = {}
for c in range(3):
    c_train, c_test, c_train_labels, c_test_labels = preprocessing.train_test_split(Data_X[Data_y==c], Data_y[Data_y==c], test_size=0.3)
    classes[c] = {
        'train_features': c_train,
        'test_features': c_test,
        'train_labels': c_train_labels,
        'test_labels': c_test_labels
    }
    
X_train = np.vstack([classes[c]['train_features'] for c in range(3)])
Y_train = np.concatenate([classes[c]['train_labels'] for c in range(3)])
X_test = np.vstack([classes[c]['test_features'] for c in range(3)])
Y_test = np.concatenate([classes[c]['test_labels'] for c in range(3)])


ovo_model = Perceptron_Multiclass_OVO(
    lr=0.01,
    epochs=1000,
    activation="logistic"
)

classifiers = ovo_model.fit(
    X_train,
    Y_train
)

Y_pred = ovo_model.predict(
    X_test
)

metrics = Metrics(Y_test, Y_pred)
metrics.classification_report()

max_length = max(len(pair['train_error']) for pair in classifiers.values())
error_collection = np.array([classifiers[pair]['train_error'] + [0] * (max_length - len(classifiers[pair]['train_error'])) for pair in classifiers])
avg_train_error = np.mean(error_collection, axis=0)

plotter = Plots()
plotter.line_plot(list(range(1, len(avg_train_error)+1)),avg_train_error, title="Average Training Error vs Epochs",
                  xlabel="Epochs",ylabel="Average Error", save_path="average_error_vs_epochs_logistic.png")

plotter.plot_cm_from_scratch(metrics.confusion_matrix(), ['class1','class2','class3'], save_path="Confusion_matrix_logistic.png")

plotter.plot_pair_wise(
    X_train,
    Y_train,
    classifiers,
    0,
    1,
    savepath="decision_region_C1_C2_logistic.png"
)

plotter.plot_pair_wise(
    X_train,
    Y_train,
    classifiers,
    0,
    2,
    savepath="decision_region_C1_C3_logistic.png"
)

plotter.plot_pair_wise(
    X_train,
    Y_train,
    classifiers,
    1,
    2,
    savepath="decision_region_C2_C3_logistic.png"
)

plotter.plot_combined_ovo_decision_region(
    X_train,
    Y_train,
    ovo_model,
    save_path="combined_ovo_decision_region_logistic.png"
)

ovo_model = Perceptron_Multiclass_OVO(
    lr=0.01,
    epochs=1000,
    activation="tanh"
)

classifiers = ovo_model.fit(
    X_train,
    Y_train
)

Y_pred = ovo_model.predict(
    X_test
)

metrics = Metrics(Y_test, Y_pred)
metrics.classification_report()

max_length = max(len(pair['train_error']) for pair in classifiers.values())
error_collection = np.array([classifiers[pair]['train_error'] + [0] * (max_length - len(classifiers[pair]['train_error'])) for pair in classifiers])
avg_train_error = np.mean(error_collection, axis=0)

plotter = Plots()
plotter.line_plot(list(range(1, len(avg_train_error)+1)),avg_train_error, title="Average Training Error vs Epochs",
                  xlabel="Epochs",ylabel="Average Error", save_path="average_error_vs_epochs_tanh.png")

plotter.plot_cm_from_scratch(metrics.confusion_matrix(), ['class1','class2','class3'], save_path="Confusion_matrix_tanh.png")

plotter.plot_pair_wise(
    X_train,
    Y_train,
    classifiers,
    0,
    1,
    savepath="decision_region_C1_C2_tanh.png"
)

plotter.plot_pair_wise(
    X_train,
    Y_train,
    classifiers,
    0,
    2,
    savepath="decision_region_C1_C3_tanh.png"
)

plotter.plot_pair_wise(
    X_train,
    Y_train,
    classifiers,
    1,
    2,
    savepath="decision_region_C2_C3_tanh.png"
)

plotter.plot_combined_ovo_decision_region(
    X_train,
    Y_train,
    ovo_model,
    save_path="combined_ovo_decision_region_tanh.png"
)