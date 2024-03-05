# deep_learning/supervised_dl/evaluation_dl.py

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
