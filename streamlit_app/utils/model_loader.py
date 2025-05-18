import joblib
from models.model import LogisticRegressionModel

def load_model():
    import torch
    torch.classes.__path__ = []

    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/features.pkl")

    input_dim = len(feature_names)
    model = LogisticRegressionModel(input_dim)
    model.load_state_dict(torch.load("models/logistic_model_weights.pth", weights_only=True))
    model.eval()

    return model, scaler, feature_names
