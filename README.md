# Zomato Restaurant Success Prediction

Predicting whether a restaurant in Bangalore will be successful based on Zomato data.

## What This Does

This project predicts if a restaurant will succeed (get a rating of 3.75 or higher) by looking at things like location, cuisine type, pricing, and whether they offer online ordering.

## The Data

Uses 51,000+ Bangalore restaurants from Zomato with info like:
- Restaurant location and type
- Cuisines offered
- Online ordering and table booking options
- Ratings and votes
- Average cost for two people

## Project Files

- `data_pipeline.py` - Cleans and prepares the data
- `utils.py` - Helper functions
- `model.py` - Machine learning model
- `train.py` - Trains and tests the model
- `inference.py` - Makes predictions
- `streamlit_app.py` - Web interface to try it out

## How to Run

```bash
# Install requirements
pip install -r requirements.txt

# Prepare the data
python src/data_pipeline.py

# Train the model
python src/train.py

# Launch web app
streamlit run src/streamlit_app.py
```
"# -zomaito_classifier" 
