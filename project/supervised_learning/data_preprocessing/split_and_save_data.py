import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def split_and_save_data(df):
    """
    Split the input DataFrame into training and testing sets, and save them to separate CSV files.

    Parameters:
    - df: The input DataFrame containing the data.

    Returns:
    - X_train: The features of the training set.
    - X_test: The features of the testing set.
    - y_train: The labels of the training set.
    - y_test: The labels of the testing set.
    """
    # Split the data
    X = df.drop(columns='label')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create directories for saving train and test data if they don't exist

    os.makedirs('...', exist_ok=True)
    os.makedirs('...', exist_ok=True)

    # Concatenate features and labels for train and test sets
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    # Save train and test data to separate CSV files
    train_data.to_csv('...', index=False)
    test_data.to_csv('...', index=False)

    # Return the split sets
    return X_train, X_test, y_train, y_test

