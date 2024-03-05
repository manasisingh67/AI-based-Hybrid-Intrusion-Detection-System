# deep_learning/supervised_dl/saving_dl.py

def save_model(model, model_path):
    """Save the trained model to a file."""
    model.save(model_path)
    print(f'Model saved to {model_path}')
