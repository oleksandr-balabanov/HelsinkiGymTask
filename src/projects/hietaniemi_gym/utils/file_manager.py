import os

def get_data_save_dir(base_path="./artifacts/hietaniemi_gym/0.0.1/data", dir_name="imgs_analysis"):
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