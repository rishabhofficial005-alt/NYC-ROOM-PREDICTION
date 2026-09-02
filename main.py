import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field

class Features(BaseModel):
    latitude : float = Field(...,ge=-90,le=90,description='Latitude must be between the -90 and 90')
    longitude : float = Field(...,ge=-180,le=180,description='Longitude must be between the -180 and 180')
    price: float = Field(...,ge=0,description='Price per night ,must be Positive')
    minimum_nights: int = Field(...,ge=0,description='Minimum nights requored for booking')
    number_of_reviews: int = Field(...,ge=0,description='Total Number of Reviews')
    reviews_per_month: float = Field(...,ge=0,description='Average reviewsm per night')
    calculated_host_listings_count: int =Field(...,ge=0,description='Number of Listings by the host ')
    availability_365: int = Field(...,ge=0,le=365, description='Days Available out of 365')
    neighbourhood_group: str= Field(...,min_length=1,description='Borough or neighbourhood group')
    neighbourhood: str= Field(...,min_length=1,description='Specific neighbourhood name')
model=joblib.load('Model_Classification.pkl') ##Load the pretrained model
COLUMNS=['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365']
app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def greet():
    return FileResponse("index.html")

@app.post('/predict')
def predict(features:Features):
    row=pd.DataFrame([features.dict()],columns=COLUMNS)
    prediction=model.predict(row)
    probability=model.predict_proba(row)
    return {
        "Predicted_room_type": prediction.tolist(),
        "Probability":probability.tolist()
    }


