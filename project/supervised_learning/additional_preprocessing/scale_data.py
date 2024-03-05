# additional_preprocessing/scale_data.py

from sklearn.preprocessing import MinMaxScaler

def scale_data(X_train, X_test):
    """
    Apply MinMaxScaler to the training set and transform the test set.

    Parameters:
    - X_train: Training set features.
    - X_test: Test set features.

    Returns:
    - X_train_scaled: Scaled training set features.
    - X_test_scaled: Scaled test set features.
    """
    scaler = MinMaxScaler()

    # Fit and transform the training set
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test set using the same scaler
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled
