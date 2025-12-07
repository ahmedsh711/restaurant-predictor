import streamlit as st
import pickle
import pandas as pd
import os

st.set_page_config(page_title="Restaurant Success Predictor")

@st.cache_resource
def load_stuff():
    model_path = 'models/restaurant_model.pkl'
    features_path = 'data/preprocessed/feature_names.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(features_path, 'rb') as f:
            features = pickle.load(f)
        return model, features
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.info("Make sure 'models/restaurant_model.pkl' and 'data/preprocessed/feature_names.pkl' exist")
        st.stop()

model, feature_names = load_stuff()

st.title("Restaurant Success Predictor")
st.write("Enter restaurant details to predict success probability")

st.markdown("---")

st.subheader("Basic Information")

col1, col2 = st.columns(2)

with col1:
    online = st.radio("Online ordering?", ["Yes", "No"])
    booking = st.radio("Table booking?", ["Yes", "No"])

with col2:
    cost = st.slider("Cost for two people (Rupees)", 100, 3000, 800, step=50)
    votes = st.number_input("Number of reviews", 0, 10000, 500, step=50)

st.subheader("Location")

col1, col2 = st.columns(2)

with col1:
    location_freq = st.slider("Restaurants in neighborhood", 100, 5000, 1500, step=100)

with col2:
    city_freq = st.slider("Restaurants in city zone", 500, 10000, 3000, step=100)

st.subheader("Cuisine Types")

cuisine_count = st.number_input("How many cuisine types?", 1, 10, 3)

col1, col2, col3 = st.columns(3)

with col1:
    north_indian = st.checkbox("North Indian")
    chinese = st.checkbox("Chinese")
    
with col2:
    south_indian = st.checkbox("South Indian")
    continental = st.checkbox("Continental")
    
with col3:
    fast_food = st.checkbox("Fast Food")
    pizza = st.checkbox("Pizza")

if st.button("Predict Success", type="primary", use_container_width=True):
    
    data = pd.DataFrame(0, index=[0], columns=feature_names)
    
    data['online_order'] = 1 if online == "Yes" else 0
    data['book_table'] = 1 if booking == "Yes" else 0
    data['votes'] = votes
    data['cost_for_two'] = cost
    data['location_freq'] = location_freq
    data['city_freq'] = city_freq
    data['cuisine_count'] = cuisine_count
    
    cuisine_mapping = {
        'cuisine_north_indian': north_indian,
        'cuisine_south_indian': south_indian,
        'cuisine_chinese': chinese,
        'cuisine_continental': continental,
        'cuisine_fast_food': fast_food,
        'cuisine_pizza': pizza
    }
    
    for cuisine_col, is_checked in cuisine_mapping.items():
        if is_checked and cuisine_col in data.columns:
            data[cuisine_col] = 1
    
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if prediction == 1:
            st.success("This restaurant will likely succeed")
        else:
            st.error("This restaurant might struggle")
    
    with col2:
        st.metric("Success Probability", f"{probability:.0%}")
    
    st.progress(probability)
    
    st.subheader("Recommendations")
    
    recommendations = []
    
    if data['online_order'].values[0] == 0:
        recommendations.append("Consider adding online ordering")
    
    if data['book_table'].values[0] == 0:
        recommendations.append("Table booking can improve customer experience")
    
    if votes < 200:
        recommendations.append(f"Focus on getting more customer reviews (current: {votes})")
    
    if cuisine_count < 2:
        recommendations.append("Adding cuisine variety might attract more customers")
    
    if cost > 2000:
        recommendations.append("High price point - ensure premium experience matches cost")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("Good configuration! Keep maintaining quality.")

st.markdown("---")
st.caption("Restaurant Success Prediction Model")