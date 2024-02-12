import pandas as pd
import logging
import copy
from src.projects.hietaniemi_gym.data.data_consts import DEVICE_COLUMNS, TIME_COL_NAME

def aggregate_hourly_usage(dataframe: pd.DataFrame, time_col: str = TIME_COL_NAME) -> pd.DataFrame:
    """
    Aggregate the time series data to hourly frequency.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame with a time column.
    time_col (str): The name of the column in dataframe that contains time data.

    Returns:
    pd.DataFrame: The DataFrame aggregated to hourly frequency.
    """
    try:
        # create copy 
        dataframe_hourly = copy.deepcopy(dataframe)

        # Convert time column to datetime if not already
        dataframe_hourly[time_col] = pd.to_datetime(dataframe_hourly[time_col])

        # Set the time column as the DataFrame index
        dataframe_hourly.set_index(time_col, inplace=True)

        # Resample and aggregate data to hourly frequency
        dataframe_hourly = dataframe_hourly.resample('H').sum()

        # Reset index to move 'time' back to a column
        dataframe_hourly.reset_index(inplace=True)

        logging.info("Data aggregated to hourly frequency successfully.")
        return dataframe_hourly
    except Exception as e:
        logging.error(f"Error during aggregation: {e}")
        raise


def add_weekday_feature(dataframe: pd.DataFrame, time_col: str = TIME_COL_NAME) -> pd.DataFrame:
    """
    Adds a 'weekday' column to the DataFrame representing the day of the week as a number.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the gym data.
    time_col (str): The column name in dataframe that contains time data.

    Returns:
    pd.DataFrame: The DataFrame with the new 'weekday' feature added.
    """
    dataframe['weekday'] = pd.to_datetime(dataframe[time_col], utc=True).dt.weekday
    return dataframe


def add_hour_feature(dataframe: pd.DataFrame, time_col: str = TIME_COL_NAME) -> pd.DataFrame:
    """
    Adds an 'hour' column to the DataFrame representing the hour of the day.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the gym data.
    time_col (str): The column name in dataframe that contains time data.

    Returns:
    pd.DataFrame: The DataFrame with the new 'hour' feature added.
    """
    dataframe['hour'] = pd.to_datetime(dataframe[time_col], utc=True).dt.hour
    return dataframe

def add_sum_minutes_feature(dataframe: pd.DataFrame, device_columns: list = DEVICE_COLUMNS) -> pd.DataFrame:
    """
    Adds a 'sum_minutes' column to the DataFrame which is the sum of all specified device columns.

    Parameters:
    dataframe (pd.DataFrame): The dataframe containing the gym data.
    device_columns (list): List of column names representing different devices.

    Returns:
    pd.DataFrame: The DataFrame with the new 'sum_minutes' feature added.
    """
    dataframe['sum_minutes'] = dataframe[device_columns].sum(axis=1)
    return dataframe

