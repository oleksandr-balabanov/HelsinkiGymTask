import joblib
import logging
from dataclasses import dataclass

@dataclass
class PredictorModelConfig():
    model_path:str = "C:\\ML_projecsts\\Tietoevry\\artifacts\\hietaniemi_gym\\0.0.1\\models\\model.pkl"
    

class PredictorModel:
    def __init__(self, config:PredictorModelConfig):
        self.model_path = config.model_path
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = joblib.load(self.model_path)
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def predict(self, input_features):
        try:
            prediction = self.model.predict(input_features)
            logging.info("Prediction made successfully.")
            return prediction
        except Exception as e:
            logging.error(f"Error making prediction: {e}")
            raise