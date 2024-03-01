# model_testing.py

from sklearn.metrics import accuracy_score, classification_report

def test_classifier(model, X_test, y_test):
    """
    Test a trained classifier on the test set.

    Parameters:
    - model: Trained classifier model.
    - X_test: Test set features.
    - y_test: Test set labels.

    Returns:
    - accuracy: Accuracy score of the model on the test set.
    - report: Classification report containing precision, recall, F1-score, and support.
    """
    print("Testing started for " + str(model))
    y_pred = model.predict(X_test)
    print("\nTesting completed!")

    return y_pred
