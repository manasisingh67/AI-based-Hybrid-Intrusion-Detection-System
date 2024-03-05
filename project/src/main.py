# src/main.py
import sys
sys.path.append('../')

from datetime import datetime
from supervised_learning.data_preprocessing.data_loader import load_data
from supervised_learning.data_preprocessing.split_and_save_data import split_and_save_data
from supervised_learning.data_preprocessing.data_preprocessing import data_preprocessing
from supervised_learning.additional_preprocessing import apply_pca, scale_data, encode_labels
from supervised_learning.models.model_training import train_classifier
from supervised_learning.models.model_testing import test_classifier
from supervised_learning.evaluation.evaluation_metrics import calculate_accuracy, calculate_recall, calculate_f1_score, calculate_precision, generate_classification_report, generate_confusion_matrix
from supervised_learning.evaluation.save_model import save_model_evaluation
from supervised_learning.evaluation.visualize_results import visualize_confusion_matrix
from deep_learning.supervised_dl import model_dl, training_dl, saving_dl, evaluation_dl 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

def main(learning_type):
    # Load and preprocess data
    df = load_data()
    df = data_preprocessing(df)
    print(df.head())

    # Split and save data
    X_train, X_test, y_train, y_test = split_and_save_data(df)

    # Scale data
    X_train_scaled, X_test_scaled = scale_data.scale_data(X_train, X_test)

    # Encode labels
    y_train_encoded, y_test_encoded, label_mapping = encode_labels.encode_labels(y_train, y_test)

    # Apply PCA
    X_train_pca, X_test_pca = apply_pca.apply_pca(X_train_scaled, X_test_scaled)

    if learning_type.lower() == 'ml':
        # Train ML model based on user input
        classifier_name = input("Enter the classifier name (RandomForest, KNN, etc.): ")
         # Train classifier
        if classifier_name.lower() == "bernoullinb":
            model = train_classifier(BernoulliNB(), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "randomforest":
            model = train_classifier(RandomForestClassifier(), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "knn":
            model = train_classifier(KNeighborsClassifier(), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "histgradientboosting":
            model = train_classifier(HistGradientBoostingClassifier(), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "adaboost":
            model = train_classifier(AdaBoostClassifier(), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "extratrees":
            model = train_classifier(ExtraTreesClassifier(), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "linearsvc":
            model = train_classifier(LinearSVC(), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "decisiontree":
            model = train_classifier(DecisionTreeClassifier(), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "logisticregression":
            model = train_classifier(LogisticRegression(max_iter=3000), X_train_pca, y_train_encoded)
        elif classifier_name.lower() == "sgdclassifier":
            model = train_classifier(SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=3000), X_train_pca, y_train_encoded)
        else:
            raise ValueError("Unsupported classifier name")
        # Test classifier
        y_pred = test_classifier(model, X_test_pca, y_test_encoded)
        # Evaluate metrics
        accuracy = calculate_accuracy(y_test_encoded, y_pred)
        precision = calculate_precision(y_test_encoded, y_pred)
        recall = calculate_recall(y_test_encoded, y_pred)
        f1_score = calculate_f1_score(y_test_encoded, y_pred)
        classification_report = generate_classification_report(y_test_encoded, y_pred)
        confusion_matrix_data = generate_confusion_matrix(y_test_encoded, y_pred)
        # Generate a timestamp for filename uniqueness
        timestamp = datetime.now().strftime("%d%m%Y%H%M%S")
        # Saving model performance
        save_model_evaluation(X_test_pca, y_test_encoded, y_pred, accuracy, f1_score, precision, recall, generate_classification_report, confusion_matrix_data, label_mapping, classifier_name, timestamp, output_folder='...')
        # Visualize results
        visualize_confusion_matrix(confusion_matrix_data, classifier_name, timestamp, output_folder='...')
    else:
        # Build and train a deep learning model
        input_shape = (X_train_pca.shape[1],)  # Assuming X_train_pca is your input data after PCA
        num_classes = len(label_mapping)    # Number of classes in your dataset
        # Build the model
        model = model_dl(input_shape, num_classes)
        # Train the model
        history = training_dl(model, X_train, y_train, X_test, y_test)
        # Evaluate the model
        evaluation_dl(model, X_test, y_test)
        # Save the model
        saving_dl(model, "multiclass_classification_model.h5")

if __name__ == "__main__":
    learning_type = input("Enter the learning type : ml or dl")
    main(learning_type)
