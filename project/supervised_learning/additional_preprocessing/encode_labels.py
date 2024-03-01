from sklearn.preprocessing import LabelEncoder

def encode_labels(y_train, y_test):
    """
    Encode labels using LabelEncoder and keep track of label names.

    Parameters:
    - y_train: Labels in the training set.
    - y_test: Labels in the testing set.

    Returns:
    - y_train_encoded: Encoded labels in the training set.
    - y_test_encoded: Encoded labels in the testing set.
    - label_mapping: Dictionary mapping label names to their corresponding numeric values.
    """
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode labels in the training set
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Encode labels in the testing set
    y_test_encoded = label_encoder.transform(y_test)

    # Create a dictionary mapping label names to numeric values
    label_mapping = {label: encoded_value for label, encoded_value in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

    # Assuming y_train and y_test are the original label arrays
    # Assuming label_mapping, y_train_encoded, and y_test_encoded are already computed

    # Printing the top 5 instances in the training set
    print("Top 15 instances in the training set:")
    for original_label, encoded_label in zip(y_train[:15], y_train_encoded[:15]):
        print(f"Original Label: {original_label}, Encoded Label: {encoded_label}")

    # Printing the top 5 instances in the testing set
    print("\nTop 15 instances in the testing set:")
    for original_label, encoded_label in zip(y_test[:15], y_test_encoded[:15]):
        print(f"Original Label: {original_label}, Encoded Label: {encoded_label}")

    # Printing the label mapping
    print("\nLabel Mapping:")
    for label, encoded_value in label_mapping.items():
        print(f"Original Label: {label}, Encoded Label: {encoded_value}")

    return y_train_encoded, y_test_encoded, label_mapping
