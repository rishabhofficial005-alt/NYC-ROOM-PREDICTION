🏙️ NYC Airbnb Room Type Predictor

An end-to-end Machine Learning classification application that predicts the room type of an Airbnb listing in New York City using listing, location, pricing, review, host, and availability information.

The project takes a trained Machine Learning pipeline and turns it into a usable web application through FastAPI, a custom HTML/CSS/JavaScript frontend, and Render deployment.

🌐 Live Demo

🚀 Live Application:
https://nyc-room-prediction-1-dfwq.onrender.com

The application allows users to enter Airbnb listing details and receive:

Predicted room type

Class probabilities

API connection status

Input validation and error feedback

🎯 Problem Statement

The objective is to classify an Airbnb listing into one of three room-type categories:

🏠 Entire Home / Apartment

🚪 Private Room

🛋️ Shared Room

The model uses information available from an Airbnb listing to learn patterns associated with these room types.

📊 Dataset

This project uses the New York City Airbnb Open Data dataset.

Dataset:

AB_NYC_2019.csv

The dataset contains:

48,895 rows
16 columns

The original dataset includes listing information such as:

Listing ID

Listing name

Host information

Neighbourhood group

Neighbourhood

Latitude

Longitude

Room type

Price

Minimum nights

Number of reviews

Last review

Reviews per month

Host listing count

Availability over 365 days

The notebook loads the New York City Airbnb dataset and confirms a shape of (48895, 16).

🔍 Exploratory Data Analysis

The notebook performs exploratory analysis before model training.

The analysis includes:

Dataset structure

Data types

Descriptive statistics

Missing-value inspection

Room-type distribution

Numerical feature analysis

Categorical feature analysis

Feature relationships

Correlation analysis

The dataset contains three target classes:

Entire home/apt
Private room
Shared room

🧹 Data Preparation

The project builds a preprocessing pipeline so that the same transformations used during training are automatically applied during inference.

Numerical Features

The final numerical feature set contains:

price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
longitude
latitude

Numerical preprocessing:

Numerical Features
       ↓
Median Imputation
       ↓
StandardScaler

Categorical Features

The categorical feature set contains:

neighbourhood_group
neighbourhood

Categorical preprocessing:

Categorical Features
       ↓
Most-Frequent Imputation
       ↓
OneHotEncoder
       ↓
handle_unknown="ignore"

The notebook uses a ColumnTransformer to combine these preprocessing pipelines.

🤖 Machine Learning Model

The project evaluates classification algorithms and uses a Random Forest Classifier as the final model.

The final pipeline consists of:

Input Data
    ↓
ColumnTransformer
    ├── Numerical Pipeline
    │      ├── SimpleImputer(strategy="median")
    │      └── StandardScaler()
    │
    └── Categorical Pipeline
           ├── SimpleImputer(strategy="most_frequent")
           └── OneHotEncoder(handle_unknown="ignore")
    ↓
RandomForestClassifier
    ↓
Prediction + Probability

⚙️ Hyperparameter Tuning

The Random Forest model was optimized using RandomizedSearchCV.

The search explored:

n_estimators
max_depth
min_samples_split

The search was performed using:

n_iter = 10
cv = 3
scoring = f1_macro

The selected configuration was:

RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)

The notebook confirms these final parameters.

📈 Model Performance

The final trained pipeline achieved:

Metric

Score

Accuracy

85.7%

Macro F1-Score

75.8%

The model was evaluated on a held-out test set, and a confusion matrix was used to inspect class-wise performance.

Accuracy: 0.857
F1-Score: 0.758

🛡️ Handling Class Imbalance

The target classes are not equally represented.

To reduce the effect of class imbalance, the final Random Forest classifier uses:

class_weight="balanced"

This gives relatively greater importance to underrepresented classes during training.

💾 Model Serialization

The complete trained pipeline is serialized using Joblib:

Model_Classification.pkl

The saved artifact contains the preprocessing pipeline and the trained Random Forest model.

This allows the production API to receive raw listing information and run the same preprocessing automatically before inference.

⚡ FastAPI Backend

The Machine Learning model is exposed through a FastAPI REST API.

The backend:

Receives listing information

Validates the input with Pydantic

Converts the input into a Pandas DataFrame

Loads the trained ML pipeline

Generates the prediction

Generates class probabilities

Returns a JSON response

API flow

Client
  ↓
POST /predict
  ↓
Pydantic Validation
  ↓
Pandas DataFrame
  ↓
Saved ML Pipeline
  ↓
Random Forest
  ↓
Prediction + Probability
  ↓
JSON Response

🔌 API Endpoints

Home

GET /

The root endpoint serves the web application.

Health Check

GET /health

Example:

{
  "status": "healthy"
}

Prediction

POST /predict

Request Body

{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "price": 220,
  "minimum_nights": 2,
  "number_of_reviews": 45,
  "reviews_per_month": 1.8,
  "calculated_host_listings_count": 1,
  "availability_365": 280,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}

Response

{
  "Predicted_room_type": [
    "Entire home/apt"
  ],
  "Probability": [
    [
      0.91,
      0.08,
      0.01
    ]
  ]
}

🌐 Frontend

The frontend is implemented using:

HTML5

CSS3

JavaScript

The application provides an interactive interface for submitting Airbnb listing information.

Input Sections

Location

Latitude

Longitude

Borough

Neighbourhood

Pricing & Stay

Price per night

Minimum nights

Days available per year

Reviews & Host

Total reviews

Reviews per month

Listings managed by the host

✨ Frontend Features

🎯 Interactive prediction form

✅ Client-side input validation

🧪 "Try an example" functionality

🔄 Loading state

🟢 API connection indicator

⚡ Error handling

🏠 Human-readable room-type labels

📊 Probability bars

🌃 NYC-themed interface

📱 Responsive layout

The interface maps the model classes to:

Entire home/apt → Entire Home / Apartment
Private room   → Private Room
Shared room    → Shared Room

☁️ Deployment

The application is deployed on Render.

Production Architecture

GitHub Repository
        ↓
      Render
        ↓
    FastAPI App
        ↓
Model_Classification.pkl
        ↓
Interactive Frontend

Live URL

https://nyc-room-prediction-1-dfwq.onrender.com

📂 Project Structure

NYC-Room-Prediction/
│
├── main.py
├── index.html
├── Model_Classification.pkl
├── requirements.txt
├── .gitattributes
├── Untitled3.ipynb
│
└── assets/
    └── nyc_skyline_bg.jpg

File Description

File

Description

Untitled3.ipynb

Data analysis, preprocessing, model training, tuning and evaluation

Model_Classification.pkl

Serialized preprocessing + Random Forest pipeline

main.py

FastAPI backend and API endpoints

index.html

Interactive web frontend

requirements.txt

Python dependencies

.gitattributes

Git LFS configuration

assets/

Frontend assets

🛠️ Tech Stack

Programming

Python

JavaScript

Data Science

NumPy

Pandas

Matplotlib

Seaborn

Machine Learning

Scikit-learn

Random Forest

RandomizedSearchCV

Joblib

Backend

FastAPI

Pydantic

Uvicorn

Frontend

HTML5

CSS3

JavaScript

Deployment

Git

GitHub

Git LFS

Render

📦 Installation

1. Clone the Repository

git clone https://github.com/your-username/NYC-Room-Prediction.git
cd NYC-Room-Prediction

2. Create a Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

macOS / Linux

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

▶️ Run Locally

Start the FastAPI server:

uvicorn main:app --reload

Open:

http://localhost:8000

FastAPI interactive documentation:

http://localhost:8000/docs

Health check:

http://localhost:8000/health

📋 Requirements

The production environment uses pinned versions for the main dependencies:

fastapi==0.136.1
uvicorn==0.46.0
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.0
joblib==1.4.2
pydantic==2.13.3

Pinning the Machine Learning dependencies is especially important because the serialized .pkl model was created using the scikit-learn ecosystem.

🧪 Example Input

A sample listing can be populated from the frontend using the example button.

Example:

Latitude: 40.7128
Longitude: -74.0060
Borough: Manhattan
Neighbourhood: Midtown

Price: $220
Minimum Nights: 2
Availability: 280 days

Total Reviews: 45
Reviews Per Month: 1.8
Host Listings: 1

The application then sends the values to:

POST /predict

and displays the predicted room type and probabilities.

🔐 Input Validation

The API validates incoming data using Pydantic.

Examples:

Latitude          → -90 to 90
Longitude         → -180 to 180
Price             → >= 0
Availability      → 0 to 365
Reviews           → >= 0
Neighbourhood     → Required
Borough           → Required

Invalid requests are rejected before reaching the Machine Learning model.

🔄 End-to-End Workflow

NYC Airbnb Dataset
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Feature Transformation
        ↓
Model Training
        ↓
Model Comparison
        ↓
Randomized Hyperparameter Search
        ↓
Final Random Forest
        ↓
Model Evaluation
        ↓
Joblib Serialization
        ↓
FastAPI REST API
        ↓
HTML / CSS / JavaScript Frontend
        ↓
Render Deployment

🚀 Future Improvements

Possible improvements include:

Add automated CI/CD with GitHub Actions

Containerize the application using Docker

Add model monitoring

Add MLflow experiment tracking

Add SHAP-based model explainability

Add automated model retraining

Add API authentication

Add rate limiting

Add automated tests

Add structured application logging

Add more advanced model comparison

Improve model explainability for individual predictions

📌 Key Highlights

✅ End-to-end Machine Learning project

✅ 48,895 Airbnb records

✅ Exploratory Data Analysis

✅ Numerical and categorical preprocessing

✅ Missing-value handling

✅ Standardization

✅ One-hot encoding

✅ Class imbalance handling

✅ Random Forest classification

✅ Randomized hyperparameter tuning

✅ 85.7% test accuracy

✅ 75.8% macro F1-score

✅ Probability-based predictions

✅ Serialized ML pipeline

✅ FastAPI REST API

✅ Pydantic validation

✅ Interactive frontend

✅ Git LFS for model storage

✅ Render deployment

👨‍💻 Author

Rishabh Singh

Computer Science & Machine Learning Student

⭐ Project Summary

NYC Airbnb Room Type Predictor demonstrates how a Machine Learning model can be taken from experimentation to a production-style web application.

The project brings together:

Data Analysis
+
Machine Learning
+
Hyperparameter Tuning
+
Model Serialization
+
REST API Development
+
Frontend Development
+
Cloud Deployment

The final application is publicly available at:

🚀 https://nyc-room-prediction-1-dfwq.onrender.com
