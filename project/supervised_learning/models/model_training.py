# model_training.py

def train_classifier(model, X_train, y_train):
    """
    Train a specified classifier on the scaled and balanced data.

    Parameters:
    - model: The classifier model to be trained.
    - X_train: Scaled and balanced training set features.
    - y_train: Scaled and balanced training set labels.

    Returns:
    - model: Trained classifier model.
    """
    print("Training started for " + str(model))
    model.fit(X_train, y_train)
    print("\nTraining completed!")

    return model