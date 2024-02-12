import os
import dill
import logging
from typing import Any

def get_data_dir(base_path="./artifacts/hietaniemi_gym/0.0.1/data", dir_name="default"):
    """
    Creates a 'data_save' directory within the specified base path and ensures that it exists.
    
    Parameters:
    - base_path: The base path where the 'data_save' directory should be created. Defaults to the current directory.
    - dir_name: The name of the directory to create for saving data. Defaults to 'data_save'.
    
    Returns:
    - The path to the 'data_save' directory.
    """
    # Construct the full path to the data save directory
    data_save_path = os.path.join(base_path, dir_name)
    
    # Create the directory if it does not exist
    os.makedirs(data_save_path, exist_ok=True)
    
    # Return the path to the data save directory
    return data_save_path


def save_data(data: Any, dir_path: str, filename: str = "data.pkl") -> None:
    """
    Save data to a specified directory using dill serialization.

    Parameters:
    data (Any): The data object to be serialized and saved.
    dir_path (str): The directory path where the data file will be saved.
    filename (str): The name of the file to save the data. Defaults to 'data.pkl'.

    Returns:
    None: This function does not return anything.

    Raises:
    OSError: If the directory cannot be created.
    Exception: If the data cannot be serialized or saved.
    """

    try:
        # Check if the directory exists, if not, create it
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logging.info(f"Directory created at {dir_path}")

        # Construct the full path where the data will be saved
        file_path = os.path.join(dir_path, filename)

        # Save the data using dill
        with open(file_path, 'wb') as file:
            dill.dump(data, file)
            logging.info(f"Data saved to {file_path}")

    except OSError as e:
        logging.error(f"Could not create directory {dir_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Could not save data to {file_path}: {e}")
        raise


def load_data(dir_path: str, filename: str = "data.pkl") -> Any:
    """
    Load data from a specified directory using dill deserialization.

    Parameters:
    dir_path (str): The directory path from where the data file will be loaded.
    filename (str): The name of the file to load the data from. Defaults to 'data.pkl'.

    Returns:
    Any: The data object that was deserialized from the file.

    Raises:
    FileNotFoundError: If the file does not exist.
    Exception: If the data cannot be deserialized.
    """
    # Construct the full path to the data file
    file_path = os.path.join(dir_path, filename)

    try:
        # Load the data using dill
        with open(file_path, 'rb') as file:
            data = dill.load(file)
            logging.info(f"Data loaded from {file_path}")
            return data

    except FileNotFoundError as e:
        logging.error(f"The file {file_path} does not exist: {e}")
        raise
    except Exception as e:
        logging.error(f"Could not load data from {file_path}: {e}")
        raise