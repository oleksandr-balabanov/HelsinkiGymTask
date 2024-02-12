import logging
import pandas as pd

def max_na_series(column: pd.Series) -> int:
    """
    Calculates the maximum number of consecutive NaN values in a Series.
    """
    try:
        na_mask = column.isna()
        cumsum_mask = (~na_mask).cumsum()
        na_groups = na_mask.groupby(cumsum_mask).sum()
        return na_groups.max()
    except Exception as e:
        logging.error(f"Error calculating max NA series: {e}")
        raise

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the combined DataFrame by dropping rows with NaN values.
    """
    try:
        logging.info("Cleaning the combined DataFrame.")
        df.dropna(inplace=True)
        logging.info("Data cleaned successfully.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning data: {e}")
        raise