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



def hash_and_split(df, test_size):
    """
    Split data into reproducable train and test set.
    It hashes each record then calculates it's CRC32 integer to assign an all times unique value
    then it uses this value to split the data. 

    Parameters
    ----------
    *df : DataFrame
        The DataFrame we need to split.
    *test_size : float
        Should be between 0.0 and 1.0 and represents the propotion of the data to include in the test set.

    Returns
    -------
    split : tuple, length=2
        A tuple that contains train and test Pandas indexes

    """
    dh = df.apply(lambda x: bytes(md5(x.to_string().encode()).hexdigest().encode()), axis=1).apply(crc32)
    test_idx = dh[dh < test_size * 2**32].index
    train_idx = dh[dh > test_size * 2**32].index
    return (train_idx, test_idx)



def gen_equal_length_cat_col(df, column, n_bins):
    """
    Generates an equal length categorical column from a continuous series in a frame

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



def rmse()