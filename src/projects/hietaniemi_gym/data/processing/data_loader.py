
import pandas as pd
import logging


def load_gym_data(file_path: str) -> pd.DataFrame:
    """
    Load gym data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame: The loaded gym data as a DataFrame.
    """
    try:
        gym_data_df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return gym_data_df
    except FileNotFoundError:
        logging.error(f"The file was not found at the specified path: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")


def load_weather_data(file_path: str) -> pd.DataFrame:
    """
    Loads the weather data from a CSV file and preprocesses it.
    """
    try:
        logging.info(f"Loading weather data from {file_path}")
        weather_df = pd.read_csv(file_path)
        weather_df['Hour'] += ':00:00'
        weather_df['date'] = pd.to_datetime(
            weather_df['Year'].astype(str) + '-' +
            weather_df['Month'].astype(str).str.zfill(2) + '-' +
            weather_df['Day'].astype(str).str.zfill(2) + ' ' +
            weather_df['Hour']
        )
        weather_df['date'] = weather_df['date'].dt.tz_localize('UTC')
        logging.info("Weather data loaded and preprocessed successfully.")
        return weather_df
    except Exception as e:
        logging.error(f"Error loading weather data: {e}")
        raise
