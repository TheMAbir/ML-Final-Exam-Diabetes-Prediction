import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD AND TRAIN (We train on the fly for simplicity in this exam context)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=columns)

# Preprocessing Steps
cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_missing] = df[cols_with_missing].replace(0, np.nan)
df['AgeGroup'] = pd.cut(df['Age'], bins=[20, 30, 50, 100], labels=['Young', 'Middle-Aged', 'Senior'])

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Pipeline
numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
categorical_features = ['AgeGroup']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Train Model (Best Params from previous tuning assumed)
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
pipeline.fit(X, y)

# 2. PREDICTION FUNCTION
def predict_diabetes(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age):
    input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    input_data['AgeGroup'] = pd.cut(input_data['Age'], bins=[20, 30, 50, 100], labels=['Young', 'Middle-Aged', 'Senior'])
    
    prediction = pipeline.predict(input_data)
    proba = pipeline.predict_proba(input_data)[0][1]
    
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return f"{result} (Probability: {proba:.2f})"

# 3. GRADIO INTERFACE
iface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Glucose Level"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],
    outputs="text",
    title="Diabetes Prediction System",
    description="Enter patient details to predict diabetes risk."
)

iface.launch()