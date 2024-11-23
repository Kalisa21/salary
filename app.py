from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Initialize FastAPI
app = FastAPI()

# Load the data
data = pd.read_csv('Salary Data.csv')

# Drop rows where Salary or any feature is NaN
data = data.dropna()

# Encode categorical features
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

label_encoder_education = LabelEncoder()
data['Education Level'] = label_encoder_education.fit_transform(data['Education Level'])

label_encoder_job_title = LabelEncoder()
data['Job Title'] = label_encoder_job_title.fit_transform(data['Job Title'])

# Define features and target
X = data.drop(['Salary'], axis=1)
y = data['Salary']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Request model for user input
class SalaryPredictionRequest(BaseModel):
    age: int
    gender: str
    education_level: str
    job_title: str
    years_of_experience: float

# Mapping for categorical inputs
gender_mapping = {'male': 1, 'female': 0}
education_mapping = {
    "high school": 0,
    "bachelor's": 1,
    "master's": 2,
    "doctorate": 3
}
job_title_mapping = {name.lower(): idx for idx, name in enumerate(label_encoder_job_title.classes_)}

# Prediction endpoint
@app.post("/predict/")
async def predict_salary(request: SalaryPredictionRequest):
    # Validate and preprocess input
    try:
        gender = gender_mapping[request.gender.lower()]
        education_level = education_mapping[request.education_level.lower()]
        job_title = job_title_mapping[request.job_title.lower()]
        if not (23 <= request.age <= 53):
            raise ValueError("Age must be between 23 and 53.")
        if not (0 <= request.years_of_experience <= 25):
            raise ValueError("Years of Experience must be between 0 and 25.")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input for {str(e)}.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Prepare input for prediction
    input_data = pd.DataFrame(
        [[request.age, gender, education_level, job_title, request.years_of_experience]],
        columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
    )
    input_scaled = scaler.transform(input_data)

    # Predict salary
    predicted_salary = model.predict(input_scaled)
    return {"predicted_salary": round(predicted_salary[0], 2)}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Salary Prediction API"}
