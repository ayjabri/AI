import os
import urllib
import tarfile
import pandas as pd


def fetch_online_zip_file(URL, filename, as_frame=True, overwrite=False):
    """Download a zipped file from the internetand extract it in the active folder"""
    if os.path.exists(filename) and not overwrite:
        urllib.request.urlretrieve(os.path.join(URL,filename), filename)
        file_tgz = tarfile.open(filename)
        file_tgz.extractall('.')
        file_tgz.close()
    if as_frame:
        csv_filename = os.path.splitext(filename)[0] + '.csv'
        return pd.read_csv(csv_filename)
    pass