#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 17:38:02 2023

@author: nickarrivo
"""

import pandas as pd

# Reload the dataset
training_data = pd.read_csv('/Users/nickarrivo/Downloads/training.csv')

# 1. Check for Missing Values
missing_values = training_data.isnull().sum()

# 2. Distribution of the Target Variable
target_distribution = training_data['InPlay'].value_counts(normalize=True)

# 3. Inspect Features
feature_stats = training_data.describe()

# 4. Correlations
correlations = training_data.corr()

missing_values, target_distribution, feature_stats, correlations

# Drop rows where 'SpinRate' is NaN
training_data_cleaned = training_data.dropna(subset=['SpinRate'])
# Remove rows where 'Velo' is below 80
training_data_cleaned = training_data_cleaned[training_data_cleaned['Velo'] >= 80]

# Remove fastballs with a SpinRate below 1000
training_data_cleaned = training_data_cleaned[training_data_cleaned['SpinRate'] >= 1000]

# Check the shape of the dataset after removing the outliers
training_data_cleaned.shape


# Detecting outliers using the IQR method

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return ((data < lower_bound) | (data > upper_bound))

outliers = detect_outliers_iqr(training_data_cleaned.drop(columns='InPlay'))
outliers_count = outliers.sum()

outliers_count


training_data_cleaned.to_csv('/Users/nickarrivo/Downloads/training_data_cleaned.csv')

