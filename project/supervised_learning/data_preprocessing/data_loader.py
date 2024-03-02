# data_preprocessing/data_loader.py

import os
import pandas as pd

def load_data():
    """
    Loads data from CSV files in the 'data/raw/' directory.

    Returns:
    - df: DataFrame containing the concatenated data.
    """
    # Set the path to the directory containing the CSV files

    # Access the environment variable
    data_path = 'C:/Users/Mohit Raj/Desktop/Dataset'

    # List all files in the specified directory
    file_list = os.listdir(data_path)

    # Read each CSV file into a DataFrame and store them in a list
    dfs = [pd.read_csv(os.path.join(data_path, file)) for file in file_list]

    # Concatenate the list of DataFrames into a single DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # Return the resulting DataFrame
    return df



