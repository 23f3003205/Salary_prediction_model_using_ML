# Salary_prediction_model_using_ML
This project was created as a complimentry part of internship 

**Author:** Harshit Singh Patel

## Overview
This project implements an end-to-end machine learning workflow to predict personal income levels (above or below 50K) based on census data using Python (pandas, scikit-learn). It covers feature engineering, handling missing values, model selection, evaluation, and deployment.

## Dataset
Source: [Adult Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/adult)  
Contains demographic and employment features: age, workclass, education, marital status, occupation, relationship, race, gender, capital gain/loss, hours-per-week, native-country, and income.

## Project Steps
- **Data Loading:** Loads the dataset and explores structure/summary.
- **Missing Value Handling:** Imputes missing categorical columns (`workclass`, `occupation`, `native-country`) with column modes.
- **Feature Engineering:** One-hot encodes categorical variables, label-encodes the target.
- **Train-Test Split:** Uses 80% for training, 20% for testing.
- **Model Selection:** Compares Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- **Evaluation:** Uses metrics—accuracy, precision, recall, F1-score, ROC-AUC.
- **Best Model:** Gradient Boosting achieves the highest F1-score.
- **Deployment:** Trained model, preprocessing pipeline, and label encoder are serialized using joblib.

## Results
**Gradient Boosting** achieves highest test performance:
- Accuracy: ~0.87
- Precision: ~0.80
- Recall: ~0.61
- F1-score: ~0.69
- ROC-AUC: ~0.92

## Usage
1. Install requirements:
