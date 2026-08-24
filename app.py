import streamlit as st
import pandas as pd
import joblib
import urllib.parse

# -----------------------------
# Load the trained model
# -----------------------------
model = joblib.load("coffee_model.pkl")
coffee_encoder = joblib.load("coffee_encoder.pkl")
budget_encoder = joblib.load("budget_encoder.pkl")
walkability_encoder = joblib.load("walkability_encoder.pkl")

# -----------------------------
# Load coffee shop information
# -----------------------------
shops = pd.read_csv("shops.csv")

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="CoffeeRun AI",
    page_icon="☕",
    layout="centered"
)

# -----------------------------
# Header
# -----------------------------
st.title("🏃☕ CoffeeRun AI")
st.markdown(
    "### Helping runners discover the perfect coffee stop after every run."
)

st.write(
    """
    Answer a few questions below and let our Machine Learning model
    recommend the best coffee shop based on your preferences.
    """
)

st.divider()

# -----------------------------
# User Inputs
# -----------------------------
coffee = st.selectbox(
    "☕ Favorite Coffee",
    ["Espresso", "Latte", "Cold Brew", "Drip"]
)

budget = st.selectbox(
    "💰 Budget",
    ["$", "$$", "$$$"]
)

walkability = st.selectbox(
    "🚶 Walkability",
    ["High", "Medium", "Low"]
)

rating = st.slider(
    "⭐ Minimum Rating",
    4.0,
    5.0,
    4.5,
    0.1
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("☕ Find My Coffee Shop"):

    # Encode the inputs
    coffee_encoded = coffee_encoder.transform([coffee])[0]
    budget_encoded = budget_encoder.transform([budget])[0]
    walkability_encoded = walkability_encoder.transform([walkability])[0]

    # Create dataframe for prediction
    user_input = pd.DataFrame({
        "coffee_encoded": [coffee_encoded],
        "budget_encoded": [budget_encoded],
        "walkability_encoded": [walkability_encoded],
        "rating": [rating]
    })

    # Predict shop
    prediction = model.predict(user_input)[0]

    # Calculate AI Match Score
    probabilities = model.predict_proba(user_input)[0]
    match = round(max(probabilities) * 100)

    # Look up shop details
    shop_info = shops[shops["shop"] == prediction]

    st.balloons()

    st.success(f"🏆 We recommend **{prediction}**!")

    st.metric("🤖 AI Match Score", f"{match}%")

    st.markdown("## 🤖 Why this recommendation")

    st.write(
    f"""
Our Decision Tree Machine Learning model analyzed your preferences and found that
**{prediction}** was the closest match based on:

    ☕ Coffee Preference: **{coffee}**

    💰 Budget: **{budget}**

    🚶 Walkability: **{walkability}**

    ⭐ Minimum Rating: **{rating}**
    """
    )

    if not shop_info.empty:

        info = shop_info.iloc[0]

        st.markdown("---")

        st.header(f"☕ {info['shop']}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("⭐ Rating", info["rating"])

        with col2:
            st.metric("💰 Price", info["price"])

        st.markdown("### Why you'll like it")

        st.write(info["review"])
        
        google_link = "https://www.google.com/maps/search/" + urllib.parse.quote(prediction)

        st.link_button("📍 View on Google Maps", google_link)

        st.markdown("---")

        st.info(
        "Recommendation generated using a Decision Tree Machine Learning model."
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    """
CoffeeRun AI was developed as a prototype recommendation system using
a Decision Tree Machine Learning model trained on a demonstration dataset.
Recommendations are intended for educational purposes.
"""
)
