import numpy as np

np.random.seed(42)  # For reproducibility

class Data_preparation:

    def load_data(self, filename, delimiter=","):
        return np.loadtxt(
            filename,
            delimiter=delimiter
        )
        
    # Train-test split
    # Randomly splits the dataset into training and testing sets
    def train_test_split(self, X, Y, test_size=0.2):
        num_samples = len(X)

        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        num_test = int(num_samples * test_size)

        test_indices = indices[:num_test]
        train_indices = indices[num_test:]

        X_train = X[train_indices]
        Y_train = Y[train_indices]

        X_test = X[test_indices]
        Y_test = Y[test_indices]

        return X_train, X_test, Y_train, Y_test
    
    # Train-validation-test split
    # 60% train, 20% validation, 20% test
    def train_val_test_split(self, X, Y):
        # First split:
        # 80% temporary + 20% test

        X_temp, X_test, Y_temp, Y_test = \
            self.train_test_split(
                X,
                Y,
                test_size=0.20
            )

        # Second split:
        # 75% of 80% = 60% train
        # 25% of 80% = 20% validation

        X_train, X_val, Y_train, Y_val = \
            self.train_test_split(
                X_temp,
                Y_temp,
                test_size=0.25
            )

        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    # Stratified train-validation-test split
    # Ensures that each class is represented proportionally in each split
    # This is particularly useful for imbalanced datasets
    def stratified_train_val_test_split(self, X, Y, num_classes):

        classes = {}

        for c in range(num_classes):

            class_X = X[Y == c]
            class_Y = Y[Y == c]

            # 80% temporary + 20% test
            temp_X, test_X, temp_Y, test_Y = \
                self.train_test_split(
                    class_X,
                    class_Y,
                    test_size=0.20
                )

            # 60% train + 20% validation
            train_X, val_X, train_Y, val_Y = \
                self.train_test_split(
                    temp_X,
                    temp_Y,
                    test_size=0.25
                )

            classes[c] = {
                "train_features": train_X,
                "train_labels": train_Y,

                "val_features": val_X,
                "val_labels": val_Y,

                "test_features": test_X,
                "test_labels": test_Y
            }

        X_train = np.vstack([classes[c]["train_features"] for c in range(num_classes)])

        Y_train = np.concatenate([classes[c]["train_labels"] for c in range(num_classes)])

        X_val = np.vstack([classes[c]["val_features"] for c in range(num_classes)])

        Y_val = np.concatenate([classes[c]["val_labels"] for c in range(num_classes)])

        X_test = np.vstack([classes[c]["test_features"] for c in range(num_classes)])

        Y_test = np.concatenate([classes[c]["test_labels"] for c in range(num_classes)])

        return X_train, X_val, X_test, Y_train, Y_val, Y_test
    
    ## Normalization and Standardization
    
    # Normalization scales the data to a specific range, typically [0, 1]
    def minmax_normalize(self, X_train, X_val, X_test, feature_min=0.0, feature_max=1.0):

        X_min = np.min(X_train, axis=0)
        X_max = np.max(X_train, axis=0)

        denominator = X_max - X_min

        # Avoid division by zero for constant features
        denominator = np.where(
            denominator == 0,
            1,
            denominator
        )

        X_train_scaled = (
            (X_train - X_min) / denominator
        )

        X_val_scaled = (
            (X_val - X_min) / denominator
        )

        X_test_scaled = (
            (X_test - X_min) / denominator
        )

        X_train_scaled = (
            X_train_scaled
            * (feature_max - feature_min)
            + feature_min
        )

        X_val_scaled = (
            X_val_scaled
            * (feature_max - feature_min)
            + feature_min
        )

        X_test_scaled = (
            X_test_scaled
            * (feature_max - feature_min)
            + feature_min
        )

        return X_train_scaled, X_val_scaled, X_test_scaled

    # Standardization scales the data to have a mean of 0 and a standard deviation of 1
    def standardize(self, X_train, X_val, X_test):

        mean = np.mean(
            X_train,
            axis=0
        )

        std = np.std(
            X_train,
            axis=0
        )

        # Avoid division by zero
        std = np.where(
            std == 0,
            1,
            std
        )

        X_train_scaled = (
            (X_train - mean) / std
        )

        X_val_scaled = (
            (X_val - mean) / std
        )

        X_test_scaled = (
            (X_test - mean) / std
        )

        return X_train_scaled, X_val_scaled, X_test_scaled
        
    def one_hot_encode(self, Y, num_classes=None, activation="logistic"):

        Y = np.asarray(Y, dtype=int)

        if num_classes is None:
            num_classes = np.max(Y) + 1

        encoded = np.zeros(
            (len(Y), num_classes)
        ) if activation == "logistic" else np.full(
            (len(Y), num_classes),
            -1
        )

        encoded[np.arange(len(Y)), Y] = 1

        return encoded