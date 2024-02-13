import os
import logging
from dotenv import load_dotenv 
from src.lib.logging.logger import setup_logging
from src.projects.hietaniemi_gym.data.data_processing.data_loader import (
    load_gym_data, 
    load_weather_data
)
from src.projects.hietaniemi_gym.data.data_processing.data_clean import data_clean_na
from src.projects.hietaniemi_gym.data.data_processing.data_merge_datasets import merge_datasets
from src.projects.hietaniemi_gym.data.data_processing.data_transformation import (
    aggregate_hourly_usage,
    add_weekday_feature,
    add_hour_feature,
    add_sum_minutes_feature,
)
from src.projects.hietaniemi_gym.model.model_predict.model_predictor import PredictorModelConfig, PredictorModel
from src.projects.hietaniemi_gym.model.model_predict.model_metrics import calculate_metrics
from src.projects.hietaniemi_gym.data.data_consts import DEVICE_COLUMNS, TIME_COL_NAME

def predict_pipeline(gym_data_path, weather_data_path, model_path):
    """
    Applies a pretrained model to the data to make predictions.

    """
    setup_logging()
    logging.info("Starting prediction pipeline")

    # Load data
    logging.info("Loading gym data from {}".format(gym_data_path))
    gym_data = load_gym_data(gym_data_path)
    logging.info("Loading weather data from {}".format(weather_data_path))
    weather_data = load_weather_data(weather_data_path)

    # Clean data
    logging.info("Cleaning gym data")
    gym_data_cleaned = data_clean_na(gym_data)
    logging.info("Cleaning weather data")
    weather_data_cleaned = data_clean_na(weather_data)

    # Data transformations
    logging.info("Transforming data: Aggregating to hourly usage")
    gym_hourly_data = aggregate_hourly_usage(gym_data_cleaned, TIME_COL_NAME)

    # Merge datasets
    logging.info("Merging datasets")
    merged_data = merge_datasets(weather_data_cleaned, gym_hourly_data)

    logging.info("Transforming data: Adding weekday feature")
    merged_data = add_weekday_feature(merged_data, TIME_COL_NAME)
    logging.info("Transforming data: Adding hour feature")
    merged_data = add_hour_feature(merged_data, TIME_COL_NAME)
    logging.info("Transforming data: Adding sum of minutes feature")
    merged_data = add_sum_minutes_feature(merged_data, DEVICE_COLUMNS) 
    
    
    logging.info("Making predictions on the dataset")
    predictor_model_config = PredictorModelConfig(model_path=model_path)
    predictor_model = PredictorModel(predictor_model_config)
    features = predictor_model.extract_features(merged_data)
    if not features.empty:
        logging.info("Features are extracted succesfully.")
        predictions = predictor_model.predict(features)
        gym_usage_col = 'sum_minutes'
        true_values = merged_data[gym_usage_col].values
        metrics = calculate_metrics(true_values, predictions)
        logging.info(f"Prediction metrics are: {metrics}")
    else:
        logging.info("Features are compromised.")



if __name__=='__main__':
    # load env
    app_env = 'development'
    env_file = f"config/.env.{app_env}"
    load_dotenv(dotenv_path=env_file)

    # execute pipeline
    gym_data_path = os.getenv('GYM_DATA_PATH')
    weather_data_path = os.getenv('WEATHER_DATA_PATH')
    model_path = os.getenv('MODEL_PATH')
    predict_pipeline(gym_data_path, weather_data_path, model_path)
