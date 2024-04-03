"""
Module to download and save datamfor country_data
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo 



def save_data(df:pd.DataFrame):
    """
    Function to save data into data directory

    Args:
        df [pandas dataframe]: dataframe to be saved
    """
    # locate data directory
    current_dir = os.path.dirname(__file__)
    root_file_name = "ntu-msds-sd6125"
    if root_file_name in current_dir:
        root_dir = current_dir.split(root_file_name,1)[0]
        data_rel_dir = "ntu-msds-sd6125/data/raw" # relative directory of data files
        data_dir = os.path.join(root_dir,data_rel_dir) # absolute path of data directory

    df.to_csv(os.path.join(data_dir, 'Customer-data.csv'), index_label='index')
    print(f"Data has been successfully saved to {data_dir}")

def download(id:int):
    """
    Function to download dataset from ucirepo
    
    Args:
        id [int]: id of dataset in uci repo

    Returns:
        df [pandas dataframe]: downloaded dataset in pandas dataframe
    """
    # fetch dataset 
    wholesale_customers = fetch_ucirepo(id=id) 

    # data
    X = wholesale_customers.data.features.rename(columns={'Detergents_Paper': 'Det_Paper', 'Delicassen': 'Deli'}) # shorten names 

    y = wholesale_customers.data.targets 
    print("Dataset has been successfullyd downloaded from UCI Repo")

    return pd.concat([X,y],axis=1)


if __name__ == "__main__":
    df = download(id=292)
    save_data(df)