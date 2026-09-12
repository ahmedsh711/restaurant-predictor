from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    online_order: Literal["Yes", "No"] = Field(..., description="Whether the restaurant accepts online orders")
    book_table: Literal["Yes", "No"] = Field(..., description="Whether the restaurant accepts table bookings")
    votes: int = Field(..., ge=0, description="Number of customer reviews/votes")
    cost_for_two: float = Field(..., gt=0, description="Approximate cost for two people in Rupees")
    location: str = Field(..., min_length=1, description="Restaurant location/neighborhood in Bangalore")
    listed_in_city: str = Field(..., min_length=1, description="City zone the restaurant is listed under on Zomato")
    rest_type: str = Field(..., min_length=1, description="Restaurant type (e.g. 'Quick Bites', 'Casual Dining', 'Cafe')")
    listed_in_type: str = Field(..., min_length=1, description="Listing category on Zomato (e.g. 'Delivery', 'Dine-out', 'Cafes')")
    cuisines: str = Field(..., min_length=1, description="Comma-separated cuisine types (e.g. 'North Indian, Chinese, Biryani')")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "online_order": "Yes",
                    "book_table": "No",
                    "votes": 500,
                    "cost_for_two": 800.0,
                    "location": "BTM",
                    "listed_in_city": "BTM",
                    "rest_type": "Casual Dining",
                    "listed_in_type": "Dine-out",
                    "cuisines": "North Indian, Chinese, Biryani"
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    success_probability: float = Field(..., ge=0.0, le=1.0, description="Probability that the restaurant will succeed")
    will_succeed: bool = Field(..., description="True if predicted to achieve a rating >= 3.75")
    model_version: str = Field(..., description="Version of the model used for prediction")
    correlation_id: str = Field(..., description="Unique ID for tracing this request end-to-end")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, description="List of restaurant feature sets to predict")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse] = Field(..., description="List of prediction results")