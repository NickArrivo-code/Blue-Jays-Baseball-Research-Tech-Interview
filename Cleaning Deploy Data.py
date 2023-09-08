#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:32:58 2023

@author: nickarrivo
"""
import pandas as pd

testing_data = pd.read_csv('/Users/nickarrivo/Downloads/deploy.csv')
missing_values = testing_data.isnull().sum()
missing_values
testing_data_cleaned = testing_data.dropna()

testing_data_cleaned.shape

testing_data_cleaned.to_csv('/Users/nickarrivo/Downloads/cleaned_deploy.csv')
