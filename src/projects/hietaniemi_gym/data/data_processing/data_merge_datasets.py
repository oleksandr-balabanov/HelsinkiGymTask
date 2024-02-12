import pandas as pd
import logging

def merge_datasets(weather_df: pd.DataFrame, gym_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges weather and gym data DataFrames on their datetime indices.
    """
    try:
        logging.info("Merging weather and gym datasets.")
        weather_df.set_index('time', inplace=True)
        gym_data_df.set_index('time', inplace=True)

        combined_df = pd.merge(weather_df, gym_data_df, left_index=True, right_index=True, how='inner')

        combined_df.reset_index(inplace=True)
        logging.info("Datasets merged successfully.")
        return combined_df
    except Exception as e:
        logging.error(f"Error merging datasets: {e}")
        raise