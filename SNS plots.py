#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:09:21 2023

@author: nickarrivo
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
rf_predictions_data = pd.read_csv('/Users/nickarrivo/Downloads/Random Forest Model Predictions.csv')
rf_predictions_data = rf_predictions_data.drop(columns = ['Unnamed: 0'])

# Set up a list of features
features = ['Velo', 'SpinRate', 'HorzBreak', 'InducedVertBreak']

# Plot the distribution of each feature based on Predicted_InPlay value
plt.figure(figsize=(15, 10))

for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='Predicted_InPlay', y=feature, data=rf_predictions_data)
    plt.title(f'Distribution of {feature} based on Predicted_InPlay')

plt.tight_layout()
plt.show()