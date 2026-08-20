from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    location: str = Field(..., min_length=1, description="Location of the restaurant")
    cuisine_type: str = Field(..., min_length=1, description="Type of cuisine")
    approx_cost_for_two: float = Field(..., gt=0, description="Average price of the restaurant for two people - Cost must be > 0")
    online_order: bool