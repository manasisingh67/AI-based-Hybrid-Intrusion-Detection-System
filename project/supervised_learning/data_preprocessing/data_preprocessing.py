from sklearn.preprocessing import StandardScaler
from itertools import combinations
import numpy as np

def data_preprocessing(df):
    """
    Perform cleaning and preprocessing on the given dataframe.

    Parameters:
    - df: The dataframe to be preprocessed.

    Returns:
    - df: The preprocessed and cleaned dataframe.
    """
    # Remove leading and trailing spaces in column names
    df.columns = df.columns.str.strip()

    # Set negative values to 0 in numeric columns
    num = df._get_numeric_data()
    num[num < 0] = 0

    # Drop columns with zero variance
    zero_variance_cols = [col for col in df.columns if len(df[col].unique()) == 1]
    df.drop(columns=zero_variance_cols, axis=1, inplace=True)
    print("Zero Variance Columns:", zero_variance_cols, "are dropped!!")
    print("Shape of dataframe after removing zero variance columns:", df.shape)

    # Replace infinite values with NaN and drop rows with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(df.isna().any(axis=1).sum(), "NaN valued rows dropped")
    df.dropna(inplace=True)
    print("Shape of dataframe after Removing NaN:", df.shape)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    print("Shape of dataframe after dropping duplicate rows:", df.shape)

    # Drop columns with identical values
    column_pairs = [(i, j) for i, j in combinations(df, 2) if df[i].equals(df[j])]
    identical_cols = [col_pair[1] for col_pair in column_pairs]
    df.drop(columns=identical_cols, axis=1, inplace=True)
    print("Columns with identical values:", column_pairs, "dropped!")
    print("Shape of dataframe after removing identical value columns:", df.shape)

    # Standardize column names by stripping, lowering, and replacing spaces with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

    # Display the final modified dataframe
    print("Final modified dataframe:", df.head())

    # Return the preprocessed and cleaned dataframe
    return df
