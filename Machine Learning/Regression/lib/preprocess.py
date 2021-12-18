#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Objective:
----------

This script contains the classes and functions needed for
Preprocessing housing dataset to make it ready for machine-
learning. The goal is to streamline data manipulation in
a single pipeline that can be called to fit.
The pipeline will be very useful when fine-tunning hyperparameters.


Pipeline:
---------
1- Create categorical variable from median income
2- Split into train and testing sets and save/append
  to existing files (stratify using median income category)
3- Deal with N/A values in the set
  If Numerical -> median else -> drop
4- Add three variables:
  a. bedrooms per household
  b. room per household
  c. population per household
5- Drop columns: median_income_cat, total_bedrooms, total_rooms
  population, households
6- convert categorical columns to one_hot_encoding
  Three ways to do that:
    1- OneHotEncoder
    2- pd.to_dummies
    3- patsy.dmatrix
7- features scaling (Normalize or Standardize based on
           the presence of outliers in the set)
8- Create a column transformer class to handle all columns (cat and numerical)
  in a single fit function


=========================

Created on Sat Nov 27 17:04:43 2021

@author: ayman
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin

inf = float('inf')
BINS = [0.,1.5,3.,6.,inf]
LABELS = list(range(len(BINS)-1))



class AddCategorical(BaseEstimator, TransformerMixin):
    def __init__(self, column=None, bins=None, labels=None, inplace=False):
        self.column = column
        self.bins = bins
        self.labels = labels
        self.inplace= inplace
        
    def fit(self, x,y=None):
        return self
    
    def transform(self, x,y=None):
        cat = pd.cut(x[self.column], bins=self.bins, labels=self.labels, ordered=False)
        if self.inplace:
            return pd.concat([x, cat], axis=1)
        return cat


class Transformer(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self):
        return self
    
    def transform(self, X, y=None):
        pass


if __name__ == '__main__':
    # add_cat = Categorical('median_income', BINS, LABELS, True)
    # numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # filler = SimpleImputer(strategy='median',add_indicator=True)
    pass
