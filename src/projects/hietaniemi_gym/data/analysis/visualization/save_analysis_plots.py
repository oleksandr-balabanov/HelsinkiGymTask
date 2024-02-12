from dataclasses import dataclass, field
from typing import List, Callable
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
import copy

@dataclass
class TotalDeviceUsageConfig:
    df: pd.DataFrame
    device_columns: List[str]
    dir: str = './plots'
    file_name: str = 'total_device_usage.png'
    figsize: tuple = (10, 8)

def plot_total_device_usage(config: TotalDeviceUsageConfig) -> None:
    try:
        total_usage_per_device = config.df[config.device_columns].sum()

        plt.figure(figsize=config.figsize)
        total_usage_per_device.plot(kind='bar')
        plt.title('Device Usage at Hietaniemi Gym')
        plt.xlabel('Device')
        plt.ylabel('Total Usage')
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        os.makedirs(config.dir, exist_ok=True)
        plt.savefig(os.path.join(config.dir, config.file_name))
        plt.close()
        logging.info(f"Plot saved to {os.path.join(config.dir, config.file_name)}")
    except Exception as e:
        logging.error(f"Failed to plot total device usage: {e}")


@dataclass
class MeanUsagePerHourConfig:
    df: pd.DataFrame
    device_columns: List[str]
    dir: str = './plots'
    file_name: str = 'mean_usage_per_hour.png'
    figsize: tuple = (10, 8)

def plot_mean_usage_per_hour(config: MeanUsagePerHourConfig) -> None:
    try:
        mean_usage_per_hour = config.df[config.device_columns].groupby(config.df.index.hour).mean()

        plt.figure(figsize=config.figsize)
        mean_usage_per_hour.plot(kind='bar')
        plt.title('Mean Usage of Devices by Hour')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Mean Usage')
        plt.xticks(range(0, 24), rotation=0)
        plt.tight_layout()
        
        os.makedirs(config.dir, exist_ok=True)
        plt.savefig(os.path.join(config.dir, config.file_name))
        plt.close()
        logging.info(f"Plot saved to {os.path.join(config.dir, config.file_name)}")
    except Exception as e:
        logging.error(f"Failed to plot mean usage per hour: {e}")



@dataclass
class DeviceUsageWeekdayWeekendConfig:
    df: pd.DataFrame
    categorize_day: Callable[[pd.Timestamp], str]
    device_columns: List[str] = field(default_factory=lambda: ['device1', 'device2', 'device3'])
    dir: str = './plots'
    file_name: str = 'device_usage_weekday_weekend_comparison.png'
    figsize: tuple = (10, 8)

def plot_device_usage_weekday_weekend_comparison(config: DeviceUsageWeekdayWeekendConfig) -> None:
    try:
        dataframe_copy = copy.deepcopy(config.df)
        dataframe_copy['day_type'] = dataframe_copy.index.map(config.categorize_day)
        
        weekdays_data = dataframe_copy[dataframe_copy['day_type'] == 'weekday'][config.device_columns].mean()
        weekends_data = dataframe_copy[dataframe_copy['day_type'] == 'weekend'][config.device_columns].mean()

        aggregated_data = pd.DataFrame({'Weekdays': weekdays_data, 'Weekends': weekends_data})

        plt.figure(figsize=config.figsize)
        aggregated_data.plot(kind='bar')
        plt.title('Device Usage Comparison: Weekdays vs Weekends')
        plt.xlabel('Device')
        plt.ylabel('Average Usage')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        os.makedirs(config.dir, exist_ok=True)
        plt.savefig(os.path.join(config.dir, config.file_name))
        plt.close()
        logging.info(f"Plot saved to {os.path.join(config.dir, config.file_name)}")
    except Exception as e:
        logging.error(f"Failed to plot device usage weekday vs weekend comparison: {e}")
