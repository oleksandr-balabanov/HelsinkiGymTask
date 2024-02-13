import joblib
import logging
from dataclasses import dataclass
import pandas as pd

@dataclass
class PredictorModelConfig():
    model_path:str
    

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

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts specified features from a DataFrame, converting data types as necessary.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
        pd.DataFrame: A DataFrame containing the extracted features, or None if an error occurs.
        """
        try:
            # Convert data types with exception handling for unexpected formats
            expected_format_list = [
                ('weekday', int),
                ('hour', int),
                ('Precipitation (mm)', float), 
                ('Snow depth (cm)', float), 
                ('Temperature (degC)', float)
            ]
            for column, dtype in expected_format_list:
                try:
                    df[column] = df[column].astype(dtype)
                except KeyError:
                    logging.warning(f"Column {column} not found in the DataFrame.")
                except ValueError:
                    logging.error(f"Cannot convert column {column} to {dtype}. Check data format.")
                    return None

            # Ensure the necessary columns are present
            required_columns = ['weekday', 'hour', 'Precipitation (mm)', 'Snow depth (cm)', 'Temperature (degC)']
            if not all(column in df for column in required_columns):
                missing_columns = [column for column in required_columns if column not in df]
                logging.error(f"Missing required columns: {missing_columns}")
                return None

            # Extract the features for the model prediction
            features_df = df[required_columns]

            return features_df
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None
