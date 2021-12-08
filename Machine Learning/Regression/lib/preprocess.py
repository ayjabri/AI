#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:04:43 2021

@author: ayman
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin



class Transformer(TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self):
        return self
    
    def transform(self, X, y=None):
        pass
