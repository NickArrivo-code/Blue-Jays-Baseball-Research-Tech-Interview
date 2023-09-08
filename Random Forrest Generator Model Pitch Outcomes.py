#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:14:56 2023

@author: nickarrivo
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib as plt


data = pd.read_csv('/Users/nickarrivo/Downloads/training_data_cleaned.csv')
data = data.drop(columns = ['Unnamed: 0'])

# Separate features and target variable
X = data.drop(columns=['InPlay'])
y = data['InPlay']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest classifier with default hyperparameters
rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# Predict on the test set and calculate accuracy
y_pred_rf = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

# To make predictions on a new test dataset:
test_data_new = pd.read_csv('/Users/nickarrivo/Downloads/cleaned_deploy.csv') 
test_data_new = test_data_new.drop(columns = ['Unnamed: 0'])
predictions_new = rf_classifier.predict(test_data_new)
print(predictions_new)


# Extract feature importances from the trained Random Forest model
feature_importances = rf_classifier.feature_importances_

# Create a DataFrame for the importances
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
importance_plot = importance_df_sorted.plot(kind='bar', x='Feature', y='Importance', figsize=(10, 6), legend=False)
plt.title('Feature Importance from Random Forest')
plt.ylabel('Importance')
plt.show()

test_data_new['Predicted_InPlay'] = predictions_new
test_data_with_pred = test_data_new
test_data_with_pred.to_csv('/Users/nickarrivo/Downloads/Random Forest Model Predictions.csv')
