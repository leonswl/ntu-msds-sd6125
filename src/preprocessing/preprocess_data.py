import os
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def read_data(file) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Function to read data from data directory
    
    Args:
        file [int]: specify which data file to read

    Returns:
        X [pandas dataframe]: pandas dataframe of data file

    """
    
    # locate data directory
    current_dir = os.path.dirname(__file__)
    root_file_name = "ntu-msds-sd6125"
    if root_file_name in current_dir:
        root_dir = current_dir.split(root_file_name,1)[0]
        data_rel_dir = "ntu-msds-sd6125/data/raw" # relative directory of data files
        data_dir = os.path.join(root_dir,data_rel_dir) # absolute path of data directory

    if file == 1:
        file_name = 'Country-data.csv'
        # read data file as pandas dataframe
        df = pd.read_csv(os.path.join(data_dir,file_name))
        exclude_cols = ['country'] # exclude country column
        y = None
        X = df.drop(exclude_cols,axis=1)

    elif file == 2:
        file_name = 'Customer-data.csv'
        # read data file as pandas dataframe
        df = pd.read_csv(os.path.join(data_dir,file_name))
        y = df['Region']
        X = df.iloc[:,:-1] 

    elif file == 3:
        file_name = 'Mall_Customers.csv'
        # read data file as pandas dataframe
        df = pd.read_csv(os.path.join(data_dir,file_name))
        # rename columns
        df.rename(columns={'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'Score'}, inplace=True)
        gender_mapping = {'Male': 1, 'Female': 0}
        df['Gender'] = df['Gender'].map(gender_mapping)
        y = None
        X = df.drop('CustomerID', axis =1)

    return X, y

def standard_scale(df:pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard scaling to the input DataFrame after excluding specified columns.

    Args:
        df [pandas dataframe]: Input DataFrame to be standardized.
        
    Returns:
        df_scaled [pandas dataframe]: Transformed DataFrame after applying standard scaling.
    """

    array_scaled = StandardScaler().fit_transform(df)
    return pd.DataFrame(data=array_scaled, columns= df.columns)

def pca(df:pd.DataFrame, variance:float=0.9):
    """
    Function to perform pca on dataframe
    
    Args:
        df [pandas dataframe]
        variance [float]: default is 0.9

    Returns
        df_pca [pandas dataframe]: pca values 
    """
    pca = PCA(variance)

    df_pca = pd.DataFrame(pca.fit_transform(df))
    var = pca.explained_variance_ratio_
    print(var)
    return df_pca

def save_data(df:pd.DataFrame, file:int):
    """
    Function to save data into data directory

    Args:
        df [pandas dataframe]: dataframe to be saved
        file [int]: file specification
    """
    # locate data directory
    current_dir = os.path.dirname(__file__)
    root_file_name = "ntu-msds-sd6125"
    if root_file_name in current_dir:
        root_dir = current_dir.split(root_file_name,1)[0]
        data_rel_dir = "ntu-msds-sd6125/data/processed" # relative directory of data files
        data_dir = os.path.join(root_dir,data_rel_dir) # absolute path of data directory

    if file == 1:
        file_name = 'country_data.csv'

    elif file == 2:
        file_name = 'customer_data.csv'

    elif file == 3:
        file_name = 'mall_customers_data.csv'

    df.to_csv(os.path.join(data_dir, file_name), index_label='index')

def preprocess(file:int, variance:float):
    if file is None:
        print("No file is given. Module is exiting. Please specify a file for preprocessing")
        exit()

    else:
        # read data
        X, y = read_data(file=file)
        print(X.head())
        # perform standardisation
        X_scaled = standard_scale(X)
        print(X_scaled.head())

        # perform PCA if variance is supplied
        if variance is not None:
            # perform PCA
            X_pca = pca(X_scaled,variance)

        else:
            X_pca = X_scaled.copy()

        if y is None:
            df = X_pca.copy()
        else:
            df = pd.concat([X_pca, y], axis=1)

        save_data(df,file=file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing module for datasets")
    parser.add_argument("--file", default=None, help="file number (1: Country data)",type=int)
    parser.add_argument("--variance", default=None, help="Variance for PCA", type=float)
    args = parser.parse_args()

    
    preprocess(file = args.file, variance=args.variance)