# deep_learning/supervised_dl/model_dl.py

import tensorflow as tf

def build_multiclass_model(input_shape, num_classes):
    """Build the deep learning model for multi-class classification."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Adjusted for multi-class classification
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
