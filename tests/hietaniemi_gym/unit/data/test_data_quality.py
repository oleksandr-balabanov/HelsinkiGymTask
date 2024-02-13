import pandas as pd
import os
from dotenv import load_dotenv
from src.projects.hietaniemi_gym.data.data_processing.data_loader import load_gym_data
import pytest
from pathlib import Path

@pytest.fixture(scope="module")
def hietaniemi_gym_data_df():
    # load env
    app_env = 'development'
    env_file = Path(f"config/.env.{app_env})"
    load_dotenv(dotenv_path=env_file)
    # get data path
    gym_data_path = os.getenv('GYM_DATA_PATH')
    gym_data_path = Path(os.getenv('GYM_DATA_PATH'))
    return load_gym_data(gym_data_path)

def test_row_count(hietaniemi_gym_data_df):
    row_count = len(hietaniemi_gym_data_df)
    assert row_count > 50000, f"The dataset should have more than 50,000 rows, but has {row_count}."

def test_date_range(hietaniemi_gym_data_df):
    min_date = hietaniemi_gym_data_df['time'].min()
    max_date = hietaniemi_gym_data_df['time'].max()
    assert min_date >= pd.Timestamp('2020-04-24', tz='UTC'), f"The dataset should have records from 2020-04-24, but starts from {min_date}."
    assert max_date < pd.Timestamp('2021-05-12', tz='UTC'), f"The dataset should have records up to 2021-05-11, but goes until {max_date}."

def test_positive_values(hietaniemi_gym_data_df):
    numeric_cols = hietaniemi_gym_data_df.select_dtypes(include=['number']).columns
    non_positive_values = hietaniemi_gym_data_df[hietaniemi_gym_data_df[numeric_cols] < 0]
    assert non_positive_values.empty, "There are non-positive values in the numerical columns."



