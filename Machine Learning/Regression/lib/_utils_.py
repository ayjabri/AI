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

from sklearn.base import BaseEstimator, TransformerMixin

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


def hash_one(r,w_hash=False):
    """Return the crc32 value of a single entry with the option to hash it using md5"""
    r = ''.join(r).encode()
    if w_hash:
        r = md5(r).hexdigest()
        r = bytes(r, encoding='utf-8')
    return crc32(r) & 0xffffffff

def hash_dataframe(df, columns=None, w_hash=False):
    """
    Return crc32 of pandas dataframe

    Parameters
    ----------
    df : DataFrame
        Pandas dataframe.
    columns : list|str, optional
        A column name or list of column names to operate on. The default is None.
    w_hash : bool, optional
        Hash each row using md5 before calculating crc32. The default is False.

    Returns
    -------
    results : ndarray
        An 1D array contains crc32 values. Has the same length as 
        the input DataFrame.

    """
    n = len(df)
    if columns is not None:
        tmp = df[columns].copy()
    else:
        tmp = df.copy()
    results = np.empty(n, dtype=np.int64)
    x = tmp.astype(str).values
    for i in range(n):
        results[i] = hash_one(x[i], w_hash)
    return results


def gen_equal_length_cat_col(df, column, n_bins):
    """
    Generates an equal-length categorical column
    
    Note: this function does exactly what Pandas.qcut function does.
        I wrote it before I knew there's a similar function out there
    
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
# Preprocessing - adding attributes
# =============================================================================

class NumericTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, exp):
        super().__init__()
        self.exp = exp
    def fit(self,X, y=None):
        return self
    def transform(self, X, y=None):
        orig_columns = X.columns
        results = X.eval(self.exp)
        if isinstance(results, pd.core.series.Series):
            return results.to_numpy().reshape(-1,1)
        new_columns = results.columns.difference(orig_columns)
        return results[new_columns].to_numpy()
    

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