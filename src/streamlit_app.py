import streamlit as st
import requests

# --- Page Config ---
st.set_page_config(
    page_title="Zomato Restaurant Success Predictor",
    page_icon="🍽️",
    layout="centered"
)

# --- API Config ---
API_URL = "http://localhost:8000/predict"
API_KEY = "supersecretkey"


# --- Header ---
st.title("🍽️ Zomato Restaurant Success Predictor")
st.write("Enter restaurant details to predict whether it will be successful (rated ≥ 3.75/5) on Zomato.")

st.markdown("---")

# --- Input Form ---
st.subheader("📋 Restaurant Details")

col1, col2 = st.columns(2)

with col1:
    online_order = st.radio("📱 Online Ordering?", ["Yes", "No"], index=0)
    book_table = st.radio("🪑 Table Booking?", ["Yes", "No"], index=1)

with col2:
    cost_for_two = st.slider("💰 Cost for Two (₹)", 50, 5000, 800, step=50)
    votes = st.number_input("⭐ Number of Reviews", 0, 50000, 500, step=50)

st.markdown("---")
st.subheader("📍 Location & Type")

col3, col4 = st.columns(2)

with col3:
    location = st.selectbox(
        "Location (Neighborhood)",
        options=[
            "BTM", "HSR", "Koramangala 5th Block", "JP Nagar", "Whitefield",
            "Indiranagar", "Jayanagar", "Marathahalli", "Bannerghatta Road",
            "Bellandur", "Electronic City", "Koramangala 1st Block",
            "Brigade Road", "Koramangala 7th Block", "Koramangala 6th Block",
            "Sarjapur Road", "Ulsoor", "Koramangala 4th Block", "MG Road",
            "Banashankari", "Malleshwaram", "Basavanagudi", "Rajajinagar",
            "Yelahanka", "Church Street", "Lavelle Road", "Residency Road",
            "Frazer Town", "Old Airport Road", "Brookefield"
        ],
        index=0
    )

with col4:
    listed_in_city = st.selectbox(
        "Listed In (City Zone)",
        options=[
            "BTM", "Koramangala 7th Block", "Koramangala 5th Block",
            "Koramangala 4th Block", "Koramangala 6th Block", "Jayanagar",
            "JP Nagar", "Indiranagar", "Church Street", "MG Road",
            "Brigade Road", "Whitefield", "Bannerghatta Road", "HSR",
            "Marathahalli", "Electronic City", "Sarjapur Road",
            "Bellandur", "Malleshwaram", "Ulsoor", "Frazer Town",
            "Basavanagudi", "Banashankari", "Lavelle Road", "Residency Road",
            "Brookefield", "Old Airport Road", "Rajajinagar",
            "St. Marks Road", "Koramangala 1st Block"
        ],
        index=0
    )

col5, col6 = st.columns(2)

with col5:
    rest_type = st.selectbox(
        "🏪 Restaurant Type",
        options=[
            "Quick Bites", "Casual Dining", "Cafe", "Delivery",
            "Dessert Parlor", "Takeaway, Delivery", "Casual Dining, Bar",
            "Bakery", "Beverage Shop", "Bar", "Food Court", "Sweet Shop",
            "Bar, Casual Dining", "Lounge", "Pub", "Other"
        ],
        index=0
    )

with col6:
    listed_in_type = st.selectbox(
        "📂 Listing Category",
        options=[
            "Delivery", "Dine-out", "Desserts", "Cafes",
            "Drinks & nightlife", "Buffet", "Pubs and bars"
        ],
        index=0
    )

st.markdown("---")
st.subheader("🍛 Cuisine Types")

cuisines_input = st.multiselect(
    "Select cuisines served (pick all that apply)",
    options=[
        "North Indian", "Chinese", "South Indian", "Fast Food",
        "Continental", "Biryani", "Cafe", "Desserts", "Beverages",
        "Italian", "Street Food", "Bakery", "Pizza", "Burger",
        "Seafood", "Andhra", "Ice Cream", "Mughlai", "American",
        "Asian", "Mexican", "Thai", "Japanese", "Korean",
        "Mediterranean", "Rolls", "Momos", "Kerala", "Tibetan"
    ],
    default=["North Indian"]
)

cuisines_str = ", ".join(cuisines_input) if cuisines_input else "North Indian"

st.markdown("---")

# --- Prediction ---
if st.button("🔮 Predict Success", type="primary", use_container_width=True):

    payload = {
        "online_order": online_order,
        "book_table": book_table,
        "votes": votes,
        "cost_for_two": float(cost_for_two),
        "location": location,
        "listed_in_city": listed_in_city,
        "rest_type": rest_type,
        "listed_in_type": listed_in_type,
        "cuisines": cuisines_str
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }

    with st.spinner("Analyzing restaurant..."):
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                probability = result["success_probability"]
                will_succeed = result["will_succeed"]

                st.markdown("---")
                st.subheader("📊 Prediction Results")

                col1, col2 = st.columns([2, 1])

                with col1:
                    if will_succeed:
                        st.success("✅ This restaurant is predicted to **succeed**!")
                    else:
                        st.error("⚠️ This restaurant might **struggle** to get a high rating.")

                with col2:
                    st.metric("Success Probability", f"{probability:.0%}")

                st.progress(probability)

                # --- Recommendations ---
                st.subheader("💡 Recommendations")

                recommendations = []

                if online_order == "No":
                    recommendations.append("📱 Consider adding **online ordering** — restaurants with it tend to perform better.")

                if book_table == "No" and cost_for_two > 1000:
                    recommendations.append("🪑 For your price point, **table booking** can improve customer experience.")

                if votes < 200:
                    recommendations.append(f"⭐ Focus on getting more customer reviews (currently: {votes}). More reviews build trust.")

                if len(cuisines_input) < 2:
                    recommendations.append("🍛 Adding **cuisine variety** might attract more customers.")

                if cost_for_two > 2000:
                    recommendations.append("💎 High price point — ensure the **premium experience** justifies the cost.")

                if cost_for_two < 150:
                    recommendations.append("💰 Very low price point — check if margins are sustainable.")

                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("👍 Great configuration! Focus on maintaining food quality and service.")

            elif response.status_code == 422:
                st.error("❌ Invalid input — please check your entries and try again.")
                st.json(response.json())

            elif response.status_code in (401, 403):
                st.error("🔒 Authentication failed — check the API key configuration.")

            elif response.status_code == 503:
                st.error("🔧 Model not loaded — make sure the API server has the ONNX model.")

            else:
                st.error(f"❌ Unexpected error (HTTP {response.status_code})")
                st.text(response.text)

        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to the API server.")
            st.info(
                "Make sure the FastAPI server is running in a separate terminal:\n\n"
                "```bash\n"
                "uv run uvicorn src.api:app --reload\n"
                "```"
            )
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out — the API server might be overloaded.")

# --- Footer ---
st.markdown("---")
st.caption("Zomato Restaurant Success Prediction · 48 Features · LightGBM + ONNX Runtime + FastAPI")