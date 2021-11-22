import os
import urllib
import tarfile
import pandas as pd
import numpy as np


def fetch_online_zip_file(URL, filename, as_frame=True, overwrite=False):
    """Download a zipped file from the internet and extract it in current active folder"""
    if (not os.path.exists(filename) or overwrite):
        print('getting file from the internet.....')
        urllib.request.urlretrieve(os.path.join(URL,filename), filename)
        file_tgz = tarfile.open(filename)
        file_tgz.extractall('.')
        file_tgz.close()
    if as_frame:
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        return pd.read_csv(csv_filename)
    pass


def gen_equal_length_cat_col(df, col_name, n_bins):
    """Retruns an equal length categorical column.
    Inputs
    df: [DataFrame] 
    col_name: [str] name of the column that we want to categorize. The asumption is that 
        the column contains continuous data
    n_bins: [str] number of equally spaced categories 
    returns:
        pandas Series
    """
    l = len(df)
    s_ = int(l/n_bins)
    steps_ = list(range(s_,l,s_))
    df_tmp = df.sort_values(col_name).reset_index(drop=True)[col_name]
    min_ = np.floor(df_tmp.min())
    max_ = np.ceil(df_tmp.max())
    bins = [min_] + df_tmp.iloc[steps_].round(2).tolist() + [max_]
    labels = list(range(n_bins))
    return pd.cut(df[col_name], bins, labels=labels)