# evaluation/evaluation_metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def calculate_accuracy(y_test, y_pred):
    """
    Calculate the accuracy of a classification model.

    Parameters:
    - y_test: Test labels.
    - y_pred: Predicted labels.

    Returns:
    - accuracy: Accuracy score.
    """
    return accuracy_score(y_test, y_pred)

def calculate_precision(y_test, y_pred, average='weighted'):
    """
    Calculate the precision of a classification model.

    Parameters:
    - y_test: Test labels.
    - y_pred: Predicted labels.
    - average: Type of averaging to perform ('weighted' by default).

    Returns:
    - precision: Precision score.
    """
    return precision_score(y_test, y_pred, average=average, zero_division=1)

def calculate_recall(y_test, y_pred, average='weighted'):
    """
    Calculate the recall of a classification model.

    Parameters:
    - y_test: Test labels.
    - y_pred: Predicted labels.
    - average: Type of averaging to perform ('weighted' by default).

    Returns:
    - recall: Recall score.
    """
    return recall_score(y_test, y_pred, average=average)

def calculate_f1_score(y_test, y_pred, average='weighted'):
    """
    Calculate the F1 score of a classification model.

    Parameters:
    - y_test: Test labels.
    - y_pred: Predicted labels.
    - average: Type of averaging to perform ('weighted' by default).

    Returns:
    - f1_score: F1 score.
    """
    return f1_score(y_test, y_pred, average=average)

def generate_classification_report(y_test, y_pred):
    """
    Generate a classification report for a classification model.

    Parameters:
    - y_test: Test labels.
    - y_pred: Predicted labels.

    Returns:
    - report: Classification report.
    """
    return classification_report(y_test, y_pred)

def generate_confusion_matrix(y_test, y_pred):
    """
    Generate a confusion matrix for a classification model.

    Parameters:
    - y_test: Test labels.
    - y_pred: Predicted labels.

    Returns:
    - report: Classification report.
    """
    return confusion_matrix(y_test, y_pred)
 
