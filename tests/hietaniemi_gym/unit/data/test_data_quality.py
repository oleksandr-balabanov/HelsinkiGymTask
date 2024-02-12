# src/projects/helsinki-gym/tests/test_data_quality.py

import pandas as pd
import pytest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# It's assumed that 'hietaniemi_gym_data_df' is a global variable or imported from a module
from src.projects.hietaniemi_gym.data.data_loader import load_gym_data

# Load the data for testing
hietaniemi_gym_data_df = load_gym_data('path_to_your_csv_file')

def test_row_count():
    row_count = len(hietaniemi_gym_data_df)
    assert row_count > 50000, f"The dataset should have more than 50,000 rows, but has {row_count}."

def test_date_range():
    min_date = hietaniemi_gym_data_df['time'].min()
    max_date = hietaniemi_gym_data_df['time'].max()
    assert min_date >= pd.Timestamp('2020-04-24', tz='UTC'), f"The dataset should have records from 2020-04-24, but starts from {min_date}."
    assert max_date < pd.Timestamp('2021-05-12', tz='UTC'), f"The dataset should have records up to 2021-05-11, but goes until {max_date}."

def test_positive_values():
    numeric_cols = hietaniemi_gym_data_df.select_dtypes(include=['number']).columns
    non_positive_values = hietaniemi_gym_data_df[hietaniemi_gym_data_df[numeric_cols] < 0]
    assert non_positive_values.empty, "There are non-positive values in the numerical columns."


