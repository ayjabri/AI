#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Nov  24 12:31:24 2021

@author: Ayman Jabri
"""

import os
import urllib
import tarfile
import pandas as pd
import numpy as np


from zlib import crc32
from hashlib import md5


# =============================================================================
# Functions to fetch data from a repository
# =============================================================================
def fetch_online_zip_file(URL, filename, as_frame=True, overwrite=False):
    """Download a zipped file from the internet and extract it in current folder"""
    if (not os.path.exists(filename) or overwrite):
        urllib.request.urlretrieve(os.path.join(URL,filename), filename)
        file_tgz = tarfile.open(filename)
        file_tgz.extractall('.')
        file_tgz.close()
    if as_frame:
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        return pd.read_csv(csv_filename)
    pass


# =============================================================================
# Functions to split DataFrame
# =============================================================================
def hash_and_split(df, test_size, stratify=None):
    """
    Splits the Frame into a reproducable train and test sets.
    
    It does that by: first hashes each row using MD5, then calculates the result's CRC32. 
    CRC32 is an integer in the range [2**0, 2**32] that is unique for each row. This is then used
    to split the data propotional to the required test_size.

    Parameters
    ----------
    *df : DataFrame
        The DataFrame we need to split.
    *test_size : float
        Should be between 0.0 and 1.0 and represents the propotion of the data to include in the test set.
    stratify : str
        Column name that contains the categories to stratify the split with. Default=None
        The column dtype should be a Category.
    Returns
    -------
    split : tuple, length=2
        A tuple that contains train and test Pandas indexes

    """
    dh = df.apply(lambda x: bytes(md5(x.to_string().encode()).hexdigest().encode()), axis=1).apply(crc32)
    if stratify is not None:
        dh[stratify] = df[stratify].copy()
    test_idx = dh[dh < test_size * 2**32].index
    train_idx = dh[dh > test_size * 2**32].index
    return (train_idx, test_idx)



def gen_equal_length_cat_col(df, column, n_bins):
    """
    Generates an equal-length categorical column

    Parameters
    ----------
    *df : DataFrame
        DESCRIPTION.
    *column : str
        Name of the continuous column to generate categorical column from.
    n_bins : int
        Number of slices.

    Returns
    -------
    frame : DataFrame
        A copy of the original DataFrame with the new categorical column.
        The new column's name is the same as the input one with '_cat' added 
        to it.

    """
    new_col_name = column + "_cat"
    tmp = df.copy()
    indexes = np.split(tmp.sort_values(column).index, n_bins)
    tmp[new_col_name] = None
    for i, index in enumerate(indexes):
        tmp.loc[index, new_col_name] = i
    return tmp




# =============================================================================
# Measurment Metrics
# =============================================================================
def mse(y,y_hat):
    """Mean Squared Error (MSE): gives higher weights to large errors"""
    return np.mean((y-y_hat)**2)

def rmse(y, y_hat):
    return np.sqrt(mse(y,y_hat))


def mae(y, y_hat):
    """Mean Absolute Error is less sensitive to outliers and comparable to prediction units (USD, %..etc)"""
    return np.mean(abs(y-y_hat))