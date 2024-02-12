from logger import logger  # Assuming logger.py provides a logger object

def data_analysis_pipeline(gym_data_path, weather_data_path):
    logger.info("Starting data analysis pipeline")

    # Load data
    logger.info("Loading gym data from {}".format(gym_data_path))
    gym_data = load_gym_data(gym_data_path)
    logger.info("Loading weather data from {}".format(weather_data_path))
    weather_data = load_weather_data(weather_data_path)

    # Clean data
    logger.info("Cleaning gym data")
    gym_data_cleaned = clean_data(gym_data)
    logger.info("Cleaning weather data")
    weather_data_cleaned = clean_data(weather_data)

    # Merge datasets
    logger.info("Merging datasets")
    merged_data = merge_datasets(weather_data_cleaned, gym_data_cleaned)

    # Data transformations
    logger.info("Transforming data: Aggregating to hourly usage")
    hourly_data = aggregate_hourly_usage(merged_data, 'datetime')
    logger.info("Transforming data: Adding weekday feature")
    hourly_data = add_weekday_feature(hourly_data, 'datetime')
    logger.info("Transforming data: Adding hour feature")
    hourly_data = add_hour_feature(hourly_data, 'datetime')
    logger.info("Transforming data: Adding sum of minutes feature")
    hourly_data = add_sum_minutes_feature(hourly_data, ['device_1', 'device_2', 'device_3'])  # Adjust as necessary

    # Data plotting
    logger.info("Plotting total device usage")
    plot_total_device_usage(hourly_data)
    logger.info("Plotting popularity by time of day")
    plot_popularity_by_time_of_day(hourly_data)
    logger.info("Plotting weekday vs weekend usage")
    plot_weekday_vs_weekend_usage(hourly_data)
    logger.info("Plotting impact of weather on gym usage")
    plot_weather_impact_on_usage(hourly_data)

    logger.info("Data analysis pipeline completed successfully")
    return hourly_data  # Optionally return the final DataFrame
