# deep_learning/supervised_dl/training_dl.py

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    """Train the deep learning model."""
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history
