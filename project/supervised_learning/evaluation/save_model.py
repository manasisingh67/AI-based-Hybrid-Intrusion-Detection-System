# evaluation/save_model.py

import os
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Label mapping
def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    # If the value is not found, you may choose to return a default value or raise an exception.
    return None  # or raise ValueError("Value not found in the dictionary")

# Saving model performance
def save_model_evaluation(X_test, y_test, y_pred, accuracy, f1_score, precision, recall, classification_rep, confusion_matrix_data, label_mapping, classifier_name, timestamp, output_folder):
    """
    Save the evaluation metrics, confusion matrix, and top instances with actual and predicted labels to a file.

    Parameters:
    - model: The trained classifier model.
    - X_test: Test set features.
    - y_test: True labels for the test set.
    - y_pred: Predicted labels for the test set.
    - accuracy: Accuracy of the model.
    - classification_rep: Classification report of the model.
    - confusion_matrix_data: Confusion matrix data.
    - label_mapping: Dictionary mapping label names to their corresponding numeric values.
    - classifier_name: Name of the classifier for filename (default is "classifier").
    - output_folder: Folder path to save the output files (default is "../../output/supervised").
    """

    # Formulate the output file path with the classifier name
    output_file_path = f"{output_folder}/{classifier_name}_evaluation_{timestamp}.txt"

    # Open the file for writing
    with open(output_file_path, "w") as output_file:
        # Write evaluation metrics to the file
        output_file.write(f"\nAccuracy: {accuracy:.2f}\n")
        output_file.write("Classification Report:\n")
        output_file.write(f"{classification_rep}\n")
        output_file.write(f"The Accuracy of the Model is {accuracy}\n")
        output_file.write(f"The Precision of the Model is {f1_score}\n")
        output_file.write(f"The Recall of the Model is {precision}\n")
        output_file.write(f"The F1 Score of the Model is {recall}\n")

        # Write the confusion matrix path to the file
        output_file.write(f"\nConfusion Matrix saved as {classifier_name}_confusion_matrix_{timestamp}.png\n")

        # Print the top 100 instances with actual and predicted labels using label names
        output_file.write("\nTop 50 instances with actual and predicted labels:\n")
        for i in range(50):
            actual_label_name = get_key_by_value(label_mapping, y_test[i])
            predicted_label_name = get_key_by_value(label_mapping, y_pred[i])
            output_file.write(f"Instance {i + 1}: Actual Label - {actual_label_name}, Predicted Label - {predicted_label_name}\n")

    print(f"Output saved to {output_file_path}")
