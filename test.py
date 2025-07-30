import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier # Or your best_model
import joblib

# 1. Load the dataset (make sure adult 3.csv is in the same directory)
df = pd.read_csv('adult 3.csv')

# 2. Handling Missing Values
df = df.replace('?', np.nan)
categorical_cols_with_nan = ['workclass', 'occupation', 'native-country']
for col in categorical_cols_with_nan:
    mode_value = df[col].mode()[0]
    df[col].fillna(mode_value, inplace=True)

# 3. Feature Encoding
X = df.drop('income', axis=1)
y = df['income']

le = LabelEncoder()
y = le.fit_transform(y)

categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'native-country']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
X_transformed = preprocessor.fit_transform(X)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# 5. Train the best model (using GradientBoostingClassifier as an example)
best_model = GradientBoostingClassifier(random_state=42)
best_model.fit(X_train, y_train)

# 6. Save the new, compatible .joblib files
joblib.dump(best_model, 'salary_prediction_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(le, 'label_encoder.joblib')

print("New, compatible .joblib files have been saved.")