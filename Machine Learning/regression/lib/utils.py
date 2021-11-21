from os import path
from urllib import request
import tarfile


def fetch_online_zip_file(URL, filename):
    """Download a zipped file from the internetand extract it in the active folder"""
    request.urlretrieve(path.join(URL,filename), filename)
    file_tgz = tarfile.open(filename)
    file_tgz.extractall('.')
    file_tgz.close()
    pass