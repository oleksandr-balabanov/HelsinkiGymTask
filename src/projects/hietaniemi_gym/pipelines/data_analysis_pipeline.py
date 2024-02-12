import logging
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
from src.projects.hietaniemi_gym.data.data_analysis.data_visualization.data_plot_and_save import (
    plot_total_device_usage,
    plot_mean_usage_per_hour,
    plot_device_usage_weekday_weekend_comparison,
    plot_gym_usage_vs_temperature,
    plot_mean_gym_usage_vs_temperature,
    plot_sample_count_vs_temperature,
    plot_gym_usage_vs_precipitation,
    plot_mean_gym_usage_vs_precipitation
)

from src.projects.hietaniemi_gym.utils.file_manager import get_data_save_dir
from src.projects.hietaniemi_gym.data.data_consts import DEVICE_COLUMNS, TIME_COL_NAME


def data_analysis_pipeline(gym_data_path, weather_data_path):
    setup_logging()
    logging.info("Starting data analysis pipeline")

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
    merged_data = add_sum_minutes_feature(merged_data, DEVICE_COLUMNS)  # Adjust as necessary

    # Data plotting
    save_dir = get_data_save_dir()  # Get or create the directory to save the plots

    logging.info("Plotting total device usage")
    plot_total_device_usage(merged_data, dir=save_dir)

    logging.info("Plotting mean usage per hour")
    time_col = 'time'
    plot_mean_usage_per_hour(merged_data, time_col, dir=save_dir)

    logging.info("Plotting device usage weekday vs weekend comparison")
    plot_device_usage_weekday_weekend_comparison(merged_data, categorize_day, dir=save_dir)

    logging.info("Plotting gym usage vs temperature")
    temperature_col = 'Temperature (degC)'
    gym_usage_col = 'sum_minutes'
    plot_gym_usage_vs_temperature(merged_data, temperature_col, gym_usage_col, dir=save_dir)

    logging.info("Plotting mean gym usage vs temperature")
    num_bins = 20
    plot_mean_gym_usage_vs_temperature(merged_data, temperature_col, gym_usage_col, num_bins, dir=save_dir)

    logging.info("Plotting sample count vs temperature")
    plot_sample_count_vs_temperature(merged_data, temperature_col, gym_usage_col, num_bins, dir=save_dir)

    logging.info("Plotting gym usage vs precipitation")
    precipitation_col = 'Precipitation (mm)'
    gym_usage_col = 'sum_minutes'
    plot_gym_usage_vs_precipitation(merged_data, precipitation_col, gym_usage_col, dir=save_dir)

    logging.info("Plotting mean gym usage vs precipitation")
    plot_mean_gym_usage_vs_precipitation(merged_data, precipitation_col, gym_usage_col, num_bins, dir=save_dir)


def categorize_day(day):
    if day < 5:  # Monday=0, Sunday=6
        return 'weekday'
    else:
        return 'weekend'


if __name__=='__main__':
    gym_data_path = 'notebooks\helsinki-gym\hietaniemi-gym-data.csv'
    weather_data_path = 'notebooks\helsinki-gym\kaisaniemi-weather-data.csv'
    data_analysis_pipeline(gym_data_path, weather_data_path)