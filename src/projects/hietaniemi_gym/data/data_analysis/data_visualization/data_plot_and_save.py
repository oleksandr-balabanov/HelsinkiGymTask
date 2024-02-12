import matplotlib.pyplot as plt
import pandas as pd
import logging
import numpy as np
from typing import Callable, List
from src.projects.hietaniemi_gym.data.data_consts import DEVICE_COLUMNS
import copy
import os



def plot_total_device_usage(df: pd.DataFrame, device_columns: List[str] = DEVICE_COLUMNS, dir: str = '.', img_name: str = 'total_device_usage.png') -> None:
    """
    Plot the total usage of each device at the gym and save the figure.
    """
    try:
        total_usage_per_device = df[device_columns].sum()
        plt.figure(figsize=(5, 4))
        total_usage_per_device.plot(kind='bar')
        plt.title('Device Usage at Hietaniemi Gym')
        plt.xlabel('Device')
        plt.ylabel('Total Usage')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(dir, img_name))
        logging.info("Device usage plot saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save device usage plot: {e}")
        raise

def plot_mean_usage_per_hour(dataframe: pd.DataFrame,  time_col:str, device_columns: List[str]=DEVICE_COLUMNS, dir: str = '.', img_name: str = 'mean_usage_per_hour.png') -> None:
    """
    Plots the mean usage per hour for given device columns and saves the figure.
    """
    try:
        mean_usage_per_hour = dataframe[device_columns].groupby(dataframe[time_col].dt.hour).mean()
        mean_usage_per_hour.plot(kind='bar', figsize=(5, 4))
        plt.title('Mean Usage of Devices by Hour')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Mean Usage')
        plt.xticks(range(0, 24), rotation=0)
        plt.legend(title='Device')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, img_name))
    except Exception as e:
        print(f"An error occurred while saving plot: {e}")

def plot_device_usage_weekday_weekend_comparison(
    dataframe: pd.DataFrame,
    categorize_day: Callable[[pd.Timestamp], str],
    device_columns: List[str]=DEVICE_COLUMNS,
    dir: str = '.',
    img_name: str = 'weekday_weekend_comparison.png'
) -> None:
    """
    Plots a comparison of device usage on weekdays vs weekends and saves the figure.
    """
    dataframe_copy = copy.deepcopy(dataframe)
    dataframe_copy['day_type'] = dataframe_copy.index.map(categorize_day)
    weekdays_data = dataframe_copy[dataframe_copy['day_type'] == 'weekday'][device_columns].mean()
    weekends_data = dataframe_copy[dataframe_copy['day_type'] == 'weekend'][device_columns].mean()
    aggregated_data = pd.DataFrame({'Weekdays': weekdays_data, 'Weekends': weekends_data})
    aggregated_data.plot(kind='bar', figsize=(5, 4))
    plt.title('Device Usage Comparison: Weekdays vs Weekends')
    plt.xlabel('Device')
    plt.ylabel('Average Usage')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, img_name))

def plot_gym_usage_vs_temperature(df: pd.DataFrame, temperature_col: str, gym_usage_col: str, dir: str = '.', img_name: str = 'gym_usage_vs_temperature.png') -> None:
    """
    Creates a scatter plot of gym usage against temperature and saves the figure.
    """
    try:
        temperature = df[temperature_col]
        gym_usage = df[gym_usage_col]
        plt.figure(figsize=(5, 4))
        plt.scatter(temperature, gym_usage, alpha=0.5)
        plt.title('Gym Usage vs. Temperature')
        plt.xlabel('Temperature (degC)')
        plt.ylabel('Sum of Gym Minutes')
        plt.savefig(os.path.join(dir, img_name))
        logging.info("Scatter plot of gym usage vs. temperature saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save gym usage vs. temperature plot: {e}")
        raise

def plot_mean_gym_usage_vs_temperature(df: pd.DataFrame, temperature_col: str, gym_usage_col: str, num_bins: int, dir: str = '.', img_name: str = 'mean_gym_usage_vs_temperature.png') -> None:
    """
    Creates a plot of mean gym usage against binned temperature values with error bars and saves the figure.
    """
    try:
        
        min_temperature = df[temperature_col].min()
        max_temperature = df[temperature_col].max()
        bin_width = (max_temperature - min_temperature) / num_bins
        temperature_bins = pd.cut(df[temperature_col], bins=np.arange(min_temperature, max_temperature, bin_width))

        mean_usage_per_temp = df.groupby(temperature_bins)[gym_usage_col].mean()
        std_usage_per_temp = df.groupby(temperature_bins)[gym_usage_col].std()

        bin_centers = [(bin.left + bin.right) / 2 for bin in mean_usage_per_temp.index.categories]

        plt.figure(figsize=(5, 4))
        plt.errorbar(bin_centers, mean_usage_per_temp, yerr=std_usage_per_temp, fmt='o', ecolor='g', capthick=2)

        plt.title('Mean Gym Usage vs. Temperature with Standard Deviation')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Mean Gym Usage (Minutes)')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, img_name))
        logging.info("Mean gym usage vs. temperature plot saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save mean gym usage vs. temperature plot: {e}")
        raise

def plot_sample_count_vs_temperature(df: pd.DataFrame, temperature_col: str, gym_usage_col: str, num_bins: int, dir: str = '.', img_name: str = 'sample_count_vs_temperature.png') -> None:
    """
    Creates a bar plot of the number of samples per temperature bin and saves the figure.
    """
    try:
        min_temperature = df[temperature_col].min()
        max_temperature = df[temperature_col].max()
        bin_width = (max_temperature - min_temperature) / num_bins
        temperature_bins = pd.cut(df[temperature_col], bins=np.arange(min_temperature, max_temperature, bin_width))
        
        count_per_temp = df.groupby(temperature_bins)[gym_usage_col].count()
        bin_centers = [(bin.left + bin.right) / 2 for bin in count_per_temp.index.categories]

        plt.figure(figsize=(5, 4))
        plt.bar(bin_centers, count_per_temp, width=bin_width * 0.9, alpha=0.5, color='blue')

        plt.title('Number of Samples vs. Temperature')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Number of Samples')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, img_name))
        logging.info("Bar plot of sample count vs. temperature saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save sample count vs. temperature plot: {e}")
        raise

def plot_gym_usage_vs_precipitation(df: pd.DataFrame, precipitation_col: str, gym_usage_col: str, dir: str = '.', img_name: str = 'gym_usage_vs_precipitation.png') -> None:
    """
    Creates a scatter plot to visualize the relationship between gym usage and precipitation and saves the figure.
    """
    try:
        precipitation = df[precipitation_col]
        gym_usage = df[gym_usage_col]

        plt.figure(figsize=(5, 4))
        plt.scatter(precipitation, gym_usage, alpha=0.5)
        plt.title('Gym Usage vs. Precipitation')
        plt.xlabel('Precipitation (mm)')
        plt.ylabel('Sum of Gym Minutes')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, img_name))
        logging.info("Scatter plot of gym usage vs. precipitation saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save gym usage vs. precipitation plot: {e}")
        raise

def plot_mean_gym_usage_vs_precipitation(df: pd.DataFrame, precipitation_col: str, gym_usage_col: str, num_bins: int, dir: str = '.', img_name: str = 'mean_usage_vs_precipitation.png') -> None:
    """
    Creates a plot of mean gym usage against precipitation values with error bars and saves the figure.
    """
    try:
        min_precipitation = df[precipitation_col].min()
        max_precipitation = df[precipitation_col].max()       
        bin_width = (max_precipitation - min_precipitation) / num_bins
        precipitation_bins = pd.cut(df[precipitation_col], bins=np.arange(min_precipitation, max_precipitation, bin_width))

        mean_usage = df.groupby(precipitation_bins)[gym_usage_col].mean()
        std_usage = df.groupby(precipitation_bins)[gym_usage_col].std()

        bin_centers = [(bin.left + bin.right) / 2 for bin in mean_usage.index.categories]

        plt.figure(figsize=(5, 4))
        plt.errorbar(bin_centers, mean_usage, yerr=std_usage, fmt='o', ecolor='g', capthick=2, capsize=5)

        plt.title('Mean Gym Usage vs. Precipitation with Standard Deviation')
        plt.xlabel('Precipitation (mm)')
        plt.ylabel('Mean Gym Usage (Minutes)')
        plt.tight_layout()
        plt.savefig(os.path.join(dir, img_name))
        logging.info("Mean gym usage vs. precipitation plot saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save mean gym usage vs. precipitation plot: {e}")
        raise