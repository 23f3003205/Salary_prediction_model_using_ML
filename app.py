
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model, preprocessor, and label encoder
try:
    model = joblib.load('salary_prediction_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    le = joblib.load('label_encoder.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'salary_prediction_model.joblib', 'preprocessor.joblib', and 'label_encoder.joblib' are in the same directory.")
    st.stop()

# Define the feature columns as per the original training data
feature_cols = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                'marital-status', 'occupation', 'relationship', 'race', 'gender',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary level.')

# Create input widgets for user data
age = st.slider('Age', 17, 90, 30)
workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Without-pay', 'Never-worked'])
fnlwgt = st.number_input('Final Weight', value=100000)
education = st.selectbox('Education', ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Prof-school', '5th-6th', '10th', 'Preschool', '12th', 'Doctorate'])
educational_num = st.slider('Educational-num', 1, 16, 10)
marital_status = st.selectbox('Marital Status', ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'])
occupation = st.selectbox('Occupation', ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'])
relationship = st.selectbox('Relationship', ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
gender = st.selectbox('Gender', ['Male', 'Female'])
capital_gain = st.number_input('Capital Gain', value=0)
capital_loss = st.number_input('Capital Loss', value=0)
hours_per_week = st.slider('Hours per Week', 1, 99, 40)
native_country = st.selectbox('Native Country', ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

if st.button('Predict Salary'):
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame([[age, workclass, fnlwgt, education, educational_num,
                                marital_status, occupation, relationship, race, gender,
                                capital_gain, capital_loss, hours_per_week, native_country]],
                              columns=feature_cols)

    # Preprocess the input data
    input_transformed = preprocessor.transform(input_data)
    
    # Make a prediction
    prediction_proba = model.predict_proba(input_transformed)[:, 1]
    
    # Use a threshold to classify the prediction
    prediction = (prediction_proba > 0.5).astype(int)

    # Decode the prediction using the saved label encoder
    predicted_salary_label = le.inverse_transform(prediction)[0]
    
    st.write(f'**Predicted Salary:** {predicted_salary_label}')
    