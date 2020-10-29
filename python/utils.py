import pymc3 as pm
import pandas as pd
import theano.tensor as tt
import numpy as np
import os


def load_data(dataframe=False):
    """Recipe to load data from csv

    Args:
        dataframe (bool, optional): Load as pandas dataframe - current proper usecase. Defaults to False.

    Returns:
        tt.tensor or pd.DataFrame: tt.tensor if dataframe is false, if true, pd.DataFrame
    """
    parentdir = os.getcwd()
    df = pd.read_csv(f"{parentdir}/data/model_df.csv")
    if dataframe:
        return df
    data = {k: tt.as_tensor_variable(v) for k,v in df.to_dict("list").items()}    
    return data

def standardize(v):
    """Standardize a variable with mean 0 and std 1

    Args:
        v (np.array): Variable to standardize

    Returns:
        np.array: Standardized array
    """
    return (v - v.mean())/v.std()

def nan_buffer(arr, expected_shape, dim=1):
    """Takes an array and pads it with NANs to an expected shape

    Args:
        arr (np.array): Baseline array
        expected_shape (tuple): tuple of expected shape
        dim (int, optional): TODO - only workable for dim1 right now . Defaults to 1.

    Returns:
        np.array: Padded array
    """
    if arr.shape != expected_shape:
        curr_len = arr.shape[dim]
        expected_len = expected_shape[dim]
        add_array = np.empty((expected_shape[0], expected_len - curr_len))
        add_array[:] = np.nan
        return np.append(arr, add_array, axis=dim)
    else:
        return arr