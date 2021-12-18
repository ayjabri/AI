#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools file that contains:
    1- Subclass of Pandas DataFrame with additional feature `outliers` that 
        identifies and remove the outliers from specified columns
    2- T-test 

Created on Sat Nov 27 13:10:28 2021

@author: ayman
"""

import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from scipy import stats as st





def check_3day_freq(df):
    """
    Honestly I have no idea why I wrote this function

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    pattern = np.array([1.,1.])
    size= len(df)
    if size < 3: return False
    df1 = df.diff().dropna().dt.days.values
    for i in range(size-1):
        match = all(df1[i:i+2] == pattern)
        if match: return True
    return False


class MyDataFrame(pd.core.frame.DataFrame):
    def outliers(self, columns, method="IQR", d=1.5, remove=True):
        r"""
        Identify and potentially remove outliers from a dataset.

        Must specify which column/s to apply this funciton to. Ideally value columns.

        Parameters
        ----------
        columns : str or list 
            column/s name/s.
        method : str, default='iqr'
            'iqr' filter records that are d x interquantile away from 25 percentile
            and 75 percentile 
            'std' filter records that are d standard deviations away from the mean
        d : float, default=1.5
            The distance away from 'IQR' in both directions if the method is 'iqr'
            Or the number of standard deviations away from the mean
        remove : bool, default True
            True: remove outliers and return cleaned datafram
            False: return the outliers as dataframe 
        """
        if method.lower() == "iqr":
            q25 = self[columns].quantile(0.25).item()
            q75 = self[columns].quantile(0.75).item()
            iqr = q75 - q25
            upper_bound = q75 + d * iqr
            lower_bound =q25 - d * iqr
            if not remove:
                return self[(self[columns] > upper_bound) | (self[columns] < lower_bound)]
            return self[
                (self[columns] <= upper_bound) & (self[columns] >= lower_bound)
            ]
        elif method == "std":
            std = self[columns].std()
            mu = self[columns].mean()
            z_score = abs((self[columns] - mu) / std)
            if not remove:
                return self[z_score > d]
            return self[z_score < d]
        
    
def t_test(mu,n,x_hat,Sx, alternative='two-tailed'):
    r"""
    T-test a sample mean.
    
    H0: The sample mean is not statistically different from the population
    Ha: the sample mean is different from the population
    
    Calculates the Standard Error SE=Sx/Sqrt(n), then p-value from t-distribution 

    Parameters
    ----------
    mu : float
        Mean of the poplution from which the sample is taken.
    n : int
        The size of the sample.
    x_hat : float
        The mean of the sample.
    Sx : float
        The standard deviation of the sample.
    alternative : str, optional
        DESCRIPTION. The default is 'two-tailed'.

    Returns
    -------
    Tuple
        (T statistic, p-value).

    """
    #SE = lambda Sx, n: Sx/n**0.5
    se = Sx/n**0.5
    t = (x_hat-mu)/se
    p = st.t.cdf(t, df=n-1)
    if alternative == 'two-tailed':
        p=2*(1-st.t.cdf(abs(t), df=n-1))
    elif alternative == 'more':
        p -= 1
    elif alternative =='less':
        p = p
    else:
        raise('error')
    print(f't value:{t:.4f}, p-value:{abs(p):.4f}')
    return (t,abs(p))