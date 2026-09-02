# 🏙️ NYC Airbnb Room Type Predictor

An end-to-end **Machine Learning classification application** that predicts the room type of an Airbnb listing in New York City based on listing, location, pricing, review, host, and availability information.

The project takes a trained Machine Learning pipeline and deploys it as an interactive web application using **FastAPI, HTML, CSS, JavaScript, and Render**.

🔗 **Live Demo:** https://nyc-room-prediction-1-dfwq.onrender.com

---

## 📌 Project Overview

The **NYC Airbnb Room Type Predictor** classifies Airbnb listings into one of three room categories:

* 🏠 **Entire Home / Apartment**
* 🚪 **Private Room**
* 🛋️ **Shared Room**

The application accepts listing information through a web interface, sends the data to a FastAPI backend, processes it through the trained ML pipeline, and returns the predicted room type along with class probabilities.

### End-to-End Architecture

```text
NYC Airbnb Dataset
        ↓
Exploratory Data Analysis
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Randomized Hyperparameter Search
        ↓
Random Forest Classifier
        ↓
Model Evaluation
        ↓
Joblib Model Serialization
        ↓
FastAPI REST API
        ↓
HTML / CSS / JavaScript Frontend
        ↓
Render Deployment
```

---

## 🎯 Problem Statement

Airbnb listings contain multiple characteristics such as price, location, availability, reviews, and host information.

The objective of this project is to build a supervised Machine Learning classification system capable of predicting the **room type** of an Airbnb listing using these attributes.

This project demonstrates the complete process of taking a Machine Learning model from **data analysis and experimentation to production-style deployment**.

---

## 📊 Dataset

The project uses the **New York City Airbnb Open Data** dataset.

**Dataset:** `AB_NYC_2019.csv`

The dataset contains:

* **48,895 records**
* **16 columns**

Important attributes include:

| Feature                          | Description                        |
| -------------------------------- | ---------------------------------- |
| `id`                             | Unique listing identifier          |
| `name`                           | Listing name                       |
| `host_id`                        | Host identifier                    |
| `host_name`                      | Host name                          |
| `neighbourhood_group`            | NYC borough                        |
| `neighbourhood`                  | NYC neighbourhood                  |
| `latitude`                       | Listing latitude                   |
| `longitude`                      | Listing longitude                  |
| `room_type`                      | Target variable                    |
| `price`                          | Price per night                    |
| `minimum_nights`                 | Minimum number of nights           |
| `number_of_reviews`              | Total number of reviews            |
| `last_review`                    | Date of latest review              |
| `reviews_per_month`              | Average monthly reviews            |
| `calculated_host_listings_count` | Number of listings managed by host |
| `availability_365`               | Number of available days per year  |

---

## 🔍 Exploratory Data Analysis

Before model training, the dataset was extensively analyzed to understand its structure and identify patterns.

The analysis includes:

* Dataset structure and dimensions
* Data types
* Descriptive statistics
* Missing-value analysis
* Target-class distribution
* Numerical feature analysis
* Categorical feature analysis
* Feature relationships
* Correlation analysis
* Class imbalance investigation

### Target Classes

The target variable contains three classes:

```text
Entire home/apt
Private room
Shared room
```

---

## 🧹 Data Preprocessing

A unified **Scikit-learn preprocessing pipeline** was created to ensure that the same transformations applied during training are automatically applied during prediction.

### Numerical Features

The final numerical features are:

```text
price
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
longitude
latitude
```

### Numerical Pipeline

```text
Numerical Features
        ↓
Median Imputation
        ↓
StandardScaler
```

The numerical pipeline uses:

* `SimpleImputer(strategy="median")`
* `StandardScaler()`

### Categorical Features

The categorical features are:

```text
neighbourhood_group
neighbourhood
```

### Categorical Pipeline

```text
Categorical Features
        ↓
Most-Frequent Imputation
        ↓
OneHotEncoder
        ↓
handle_unknown="ignore"
```

The categorical pipeline uses:

* `SimpleImputer(strategy="most_frequent")`
* `OneHotEncoder(handle_unknown="ignore")`

A `ColumnTransformer` combines both pipelines into a single preprocessing workflow.

---

## 🤖 Machine Learning Model

Several classification approaches were evaluated during experimentation, with **Random Forest Classifier** selected as the final model.

### Final Pipeline

```text
Raw Input
    ↓
ColumnTransformer
    ├── Numerical Pipeline
    │     ├── Median Imputation
    │     └── StandardScaler
    │
    └── Categorical Pipeline
          ├── Most-Frequent Imputation
          └── OneHotEncoder
    ↓
Random Forest Classifier
    ↓
Prediction + Class Probabilities
```

---

## ⚙️ Hyperparameter Tuning

The Random Forest model was optimized using **RandomizedSearchCV**.

The search explored parameters including:

* `n_estimators`
* `max_depth`
* `min_samples_split`

Configuration:

```text
n_iter = 10
cv = 3
scoring = f1_macro
```

### Final Model Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42
)
```

---

## 🛡️ Handling Class Imbalance

The three room-type classes are not equally represented in the dataset.

To reduce the impact of class imbalance, the final Random Forest model uses:

```python
class_weight="balanced"
```

This assigns greater importance to underrepresented classes during model training.

---

## 📈 Model Performance

The final model was evaluated on a held-out test dataset.

| Metric             |     Score |
| ------------------ | --------: |
| **Accuracy**       | **85.7%** |
| **Macro F1-Score** | **75.8%** |

### Evaluation

The model evaluation includes:

* Accuracy
* Macro F1-score
* Confusion matrix
* Class-wise prediction analysis
* Probability-based predictions

The use of **Macro F1-score** is particularly important because it evaluates performance across all classes rather than allowing the majority class to dominate the metric.

---

## 💾 Model Serialization

The complete preprocessing and Machine Learning pipeline is serialized using **Joblib**.

```text
Model_Classification.pkl
```

The saved artifact contains:

* Numerical preprocessing
* Categorical preprocessing
* Feature transformation
* Trained Random Forest model

This allows the API to receive raw user inputs without requiring the frontend to reproduce the training transformations.

---

# ⚡ FastAPI Backend

The trained model is exposed through a **FastAPI REST API**.

The backend performs the following operations:

1. Receives listing information
2. Validates input using Pydantic
3. Converts the request into a Pandas DataFrame
4. Loads the serialized ML pipeline
5. Performs preprocessing
6. Generates the prediction
7. Calculates class probabilities
8. Returns a JSON response

### API Architecture

```text
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
```

---

## 🔌 API Endpoints

### `GET /`

Serves the web application frontend.

### `GET /health`

Returns the API health status.

Example response:

```json
{
  "status": "healthy"
}
```

### `POST /predict`

Accepts Airbnb listing information and returns a room-type prediction.

#### Request

```json
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
```

#### Response

```json
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
```

---

## 🔐 Input Validation

The API uses **Pydantic** to validate incoming requests before they reach the Machine Learning model.

Examples of validation rules:

| Input          | Validation                 |
| -------------- | -------------------------- |
| Latitude       | Between -90 and 90         |
| Longitude      | Between -180 and 180       |
| Price          | Greater than or equal to 0 |
| Minimum Nights | Greater than or equal to 0 |
| Availability   | 0–365                      |
| Reviews        | Greater than or equal to 0 |
| Borough        | Required                   |
| Neighbourhood  | Required                   |

Invalid requests are rejected before model inference.

---

# 🌐 Frontend

The frontend is built using:

* **HTML5**
* **CSS3**
* **JavaScript**

It provides a simple interactive interface for submitting Airbnb listing information and displaying the prediction.

### Input Sections

#### 📍 Location

* Latitude
* Longitude
* Borough
* Neighbourhood

#### 💰 Pricing & Stay

* Price per night
* Minimum nights
* Availability per year

#### ⭐ Reviews & Host

* Total reviews
* Reviews per month
* Host listing count

---

## ✨ Frontend Features

* 🎯 Interactive prediction form
* ✅ Client-side input validation
* 🧪 Try Example functionality
* 🔄 Loading state
* 🟢 API connection indicator
* ⚡ Error handling
* 🏠 Human-readable predictions
* 📊 Probability visualization
* 🌃 NYC-themed interface
* 📱 Responsive design

### Model Class Mapping

| Model Output      | Display Label           |
| ----------------- | ----------------------- |
| `Entire home/apt` | Entire Home / Apartment |
| `Private room`    | Private Room            |
| `Shared room`     | Shared Room             |

---

# ☁️ Deployment

The application is deployed on **Render**.

### Production Architecture

```text
GitHub Repository
       ↓
     Render
       ↓
   FastAPI App
       ↓
Model_Classification.pkl
       ↓
Interactive Frontend
```

### 🚀 Live Application

**https://nyc-room-prediction-1-dfwq.onrender.com**

The deployed application provides a complete end-to-end workflow from user input to Machine Learning prediction.

---

# 📂 Project Structure

```text
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
```

### File Descriptions

| File                       | Description                                               |
| -------------------------- | --------------------------------------------------------- |
| `Untitled3.ipynb`          | EDA, preprocessing, model training, tuning and evaluation |
| `Model_Classification.pkl` | Serialized preprocessing + Random Forest pipeline         |
| `main.py`                  | FastAPI backend and API endpoints                         |
| `index.html`               | Interactive frontend                                      |
| `requirements.txt`         | Python dependencies                                       |
| `.gitattributes`           | Git LFS configuration                                     |
| `assets/`                  | Frontend assets                                           |

---

# 🛠️ Tech Stack

### Programming

* Python
* JavaScript

### Data Science

* NumPy
* Pandas
* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* RandomizedSearchCV
* Joblib

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Frontend

* HTML5
* CSS3
* JavaScript

### Deployment & Version Control

* Git
* GitHub
* Git LFS
* Render

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/NYC-Room-Prediction.git
cd NYC-Room-Prediction
```

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally

Start the FastAPI development server:

```bash
uvicorn main:app --reload
```

The application will be available at:

```text
http://localhost:8000
```

### FastAPI Interactive Documentation

```text
http://localhost:8000/docs
```

### Health Check

```text
http://localhost:8000/health
```

---

# 📋 Requirements

The production environment uses pinned versions for the primary dependencies:

```text
fastapi==0.136.1
uvicorn==0.46.0
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.6.0
joblib==1.4.2
pydantic==2.13.3
```

Pinning Machine Learning dependencies helps maintain compatibility between the environment used to create the serialized model and the environment used for inference.

---

# 🧪 Example Prediction

Example listing:

```text
Latitude:                 40.7128
Longitude:                -74.0060
Borough:                  Manhattan
Neighbourhood:            Midtown

Price:                    $220
Minimum Nights:           2
Availability:             280 days

Total Reviews:            45
Reviews Per Month:        1.8
Host Listings:            1
```

The frontend sends this information to:

```text
POST /predict
```

The API processes the input through the trained pipeline and returns:

```text
Predicted Room Type
+
Class Probabilities
```

---

# 🔄 Complete Project Workflow

```text
AB_NYC_2019 Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Numerical & Categorical Preprocessing
        ↓
Train/Test Split
        ↓
Model Comparison
        ↓
RandomizedSearchCV
        ↓
Optimized Random Forest
        ↓
Model Evaluation
        ↓
Joblib Serialization
        ↓
FastAPI REST API
        ↓
Pydantic Validation
        ↓
HTML/CSS/JavaScript Frontend
        ↓
Render Deployment
```

---

# 🚀 Future Improvements

Potential improvements include:

* [ ] Automated CI/CD using GitHub Actions
* [ ] Docker containerization
* [ ] Automated model retraining
* [ ] MLflow experiment tracking
* [ ] SHAP-based model explainability
* [ ] Model performance monitoring
* [ ] Automated unit and integration tests
* [ ] Structured application logging
* [ ] API authentication
* [ ] Rate limiting
* [ ] Advanced model comparison
* [ ] Individual prediction explainability
* [ ] Model drift detection

---

# 📌 Key Highlights

* ✅ End-to-end Machine Learning classification project
* ✅ 48,895 Airbnb records
* ✅ Exploratory Data Analysis
* ✅ Numerical and categorical preprocessing
* ✅ Missing-value handling
* ✅ Feature standardization
* ✅ One-hot encoding
* ✅ Class imbalance handling
* ✅ Random Forest classification
* ✅ Randomized hyperparameter tuning
* ✅ **85.7% test accuracy**
* ✅ **75.8% macro F1-score**
* ✅ Probability-based predictions
* ✅ Serialized Scikit-learn pipeline
* ✅ FastAPI REST API
* ✅ Pydantic input validation
* ✅ Interactive frontend
* ✅ Git LFS model storage
* ✅ Render cloud deployment

---

# 🎓 What This Project Demonstrates

This project demonstrates practical experience across the complete Machine Learning lifecycle:

**Data → Analysis → Preprocessing → Modeling → Optimization → Evaluation → Serialization → API → Frontend → Deployment**

Rather than stopping at model training, the project focuses on transforming a Machine Learning experiment into a **usable production-style application**.

---

# 👨‍💻 Author

**Rishabh Singh**

Computer Science & Machine Learning Student

---

# ⭐ Project Summary

The **NYC Airbnb Room Type Predictor** demonstrates how Machine Learning can be integrated into a complete web application.

It combines:

**Data Analysis + Machine Learning + Hyperparameter Tuning + Model Serialization + REST API Development + Frontend Development + Cloud Deployment**

### 🚀 Try the Live Application

https://nyc-room-prediction-1-dfwq.onrender.com

If you find this project useful, consider ⭐ **starring the repository**.
