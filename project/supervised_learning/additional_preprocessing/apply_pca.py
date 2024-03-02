import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def apply_pca(X_train, X_test):
    """
    Apply PCA to the scaled features and plot explained variance ratio.

    Parameters:
    - X_train: Scaled training set features.
    - X_test: Scaled test set features.

    Returns:
    - X_train_pca: Transformed training set features after PCA.
    - X_test_pca: Transformed test set features after PCA.
    """
    pca = PCA()

    # Fit and transform the training set
    X_train_pca = pca.fit_transform(X_train)

    # Plot the explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
    plt.title('Explained Variance Ratio - Cumulative')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    # Save the variance ratio as an image file
    plt.savefig('../../images/explained_variance.png')

    # Add a comment indicating that the countplot image has been saved
    print("Explained variance ratio plot saved as 'explained_variance.png'")

    # Determine the optimal number of components (capturing 95% of the variance)
    n_components_pca = len(pca.explained_variance_ratio_[pca.explained_variance_ratio_.cumsum() < 0.95])
    print("The number of principal components explaining 95% of information:", n_components_pca)

    # Apply PCA with the selected number of components
    pca = PCA(n_components=n_components_pca)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca