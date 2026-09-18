import numpy as np
def train_test_split(X,Y, test_size=0.2):
    
    """
    Splits the dataset into training and testing sets.

    Parameters:
    X (numpy.ndarray): Input features.
    Y (numpy.ndarray): Target values.
    test_size (float): Proportion of the dataset to include in the test split.

    Returns:
    X_train, X_test, Y_train, Y_test: Split datasets.
    """
    # Calculate the number of test samples
    np.random.seed(42)  # Set a random seed for reproducibility
    num_samples = len(X)
    num_test_samples = int(num_samples * test_size)

    # Shuffle the data
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # Split the data into training and testing sets
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    X_train = X[train_indices]
    Y_train = Y[train_indices]
    X_test = X[test_indices]
    Y_test = Y[test_indices]

    return X_train, X_test, Y_train, Y_test