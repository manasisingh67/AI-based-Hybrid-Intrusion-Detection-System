# evaluation/visualize_results.py

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_confusion_matrix(confusion_matrix_data, classifier_name, timestamp, output_folder):
    """
    Visualize and save the confusion matrix plot.

    Parameters:
    - confusion_matrix_data: Confusion matrix data.
    - output_folder: Folder path to save the output files.
    - classifier_name: Name of the classifier for filename.
    - timestamp: Timestamp for filename uniqueness.
    """
    plt.figure(figsize=(15, 15))
    sns.heatmap(confusion_matrix_data, annot=True)
    plt.savefig(f"{output_folder}/{classifier_name}_confusion_matrix_{timestamp}.png")
    plt.close()
