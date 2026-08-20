from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    online_order: Literal["Yes", "No"] = Field(
        ..., description="Whether the restaurant accepts online orders"
    )
    book_table: Literal["Yes", "No"] = Field(
        ..., description="Whether the restaurant accepts table bookings"
    )
    votes: int = Field(
        ..., ge=0, description="Number of customer reviews/votes"
    )
    cost_for_two: float = Field(
        ..., gt=0, description="Approximate cost for two people in Rupees"
    )
    location: str = Field(
        ..., min_length=1, description="Restaurant location/neighborhood in Bangalore"
    )
    listed_in_city: str = Field(
        ..., min_length=1, description="City zone the restaurant is listed under on Zomato"
    )
    rest_type: str = Field(
        ..., min_length=1, description="Restaurant type (e.g. 'Quick Bites', 'Casual Dining', 'Cafe')"
    )
    listed_in_type: str = Field(
        ..., min_length=1, description="Listing category on Zomato (e.g. 'Delivery', 'Dine-out', 'Cafes')"
    )
    cuisines: str = Field(
        ..., min_length=1, description="Comma-separated cuisine types (e.g. 'North Indian, Chinese, Biryani')"
    )