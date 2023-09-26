#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:46:26 2023

@author: nickarrivo
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv('/Users/nickarrivo/Downloads/training_data_cleaned.csv')

# Extract the features Horz Break and IVB
features = ['HorzBreak', 'InducedVertBreak']
data_subset = data[features]

# Handle missing values and scale the data
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data_subset)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Perform K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
data['Cluster'] = clusters

data.to_csv('/Users/nickarrivo/Downloads/cluster pitch type.csv')


plt.figure(figsize=(10, 6))

# Using a color map to assign colors based on the Cluster column
colors = plt.cm.jet(data_subset["Cluster"] / data_subset["Cluster"].nunique())

# Creating the scatterplot
plt.scatter(data_subset["HorzBreak"], data_subset["InducedVertBreak"], c=colors, s=50, edgecolor='k')

# Adding title and labels
plt.title("Scatterplot of HorzBreak vs InducedVertBreak")
plt.xlabel("HorzBreak")
plt.ylabel("InducedVertBreak")

# Adding a colorbar to indicate cluster values
cbar = plt.colorbar()
cbar.set_label('Cluster')

plt.grid(True)
plt.show()
