import streamlit as st
import pandas as pd
import joblib
import urllib.parse

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="CoffeeRun AI",
    page_icon="☕",
    layout="centered"
)

# -----------------------------
# Load Model and Encoders
# -----------------------------
model = joblib.load("coffee_model.pkl")
coffee_encoder = joblib.load("coffee_encoder.pkl")
budget_encoder = joblib.load("budget_encoder.pkl")
walkability_encoder = joblib.load("walkability_encoder.pkl")

# -----------------------------
# Load Shop Information
# -----------------------------
shops = pd.read_csv("shops.csv")

# Clean column names
shops.columns = shops.columns.str.strip().str.lower()

# -----------------------------
# Header
# -----------------------------
st.title("🏃☕ CoffeeRun AI")

st.subheader(
    "Helping runners discover the perfect coffee stop after every run."
)

st.write(
    """
Welcome to CoffeeRun AI!

Enter your coffee preferences below and our Machine Learning model
will recommend the best coffee shop for you.
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
    min_value=4.0,
    max_value=5.0,
    value=4.5,
    step=0.1
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("☕ Find My Coffee Shop"):

    try:
        # -----------------------------
        # Encode User Inputs
        # -----------------------------
        coffee_encoded = coffee_encoder.transform([coffee])[0]
        budget_encoded = budget_encoder.transform([budget])[0]
        walkability_encoded = walkability_encoder.transform([walkability])[0]

        # -----------------------------
        # Create Model Input
        # -----------------------------
        user_input = pd.DataFrame({
            "coffee_encoded": [coffee_encoded],
            "budget_encoded": [budget_encoded],
            "walkability_encoded": [walkability_encoded],
            "rating": [rating]
        })

        # -----------------------------
        # Machine Learning Prediction
        # -----------------------------
        prediction_encoded = model.predict(user_input)[0]

        # Convert prediction back to coffee shop name
        prediction = coffee_encoder.inverse_transform([prediction_encoded])[0]

        # -----------------------------
        # Match Score
        # -----------------------------
        probabilities = model.predict_proba(user_input)[0]
        match = round(max(probabilities) * 100)

        # -----------------------------
        # Find Predicted Shop
        # -----------------------------
        if "shop" not in shops.columns:
            st.error(
                "The shops.csv file must contain a column named 'shop'."
            )
            st.stop()

        shop_info = shops[
            shops["shop"].astype(str).str.strip().str.lower()
            == str(prediction).strip().lower()
        ]

        # -----------------------------
        # If Exact Shop Isn't Found
        # -----------------------------
        if shop_info.empty:

            # Try matching using the prediction as a string
            possible_matches = shops[
                shops["shop"].astype(str).str.contains(
                    str(prediction),
                    case=False,
                    na=False
                )
            ]

            if not possible_matches.empty:
                shop_info = possible_matches

        # -----------------------------
        # Celebration
        # -----------------------------
        st.balloons()

        st.success(
            f"🏆 CoffeeRun AI recommends **{prediction}**!"
        )

        st.metric(
            "🤖 AI Match Score",
            f"{match}%"
        )

        # -----------------------------
        # Explanation
        # -----------------------------
        st.markdown("## 🤖 Why this recommendation")

        st.write(
            f"""
Our Decision Tree Machine Learning model analyzed your preferences:

- ☕ Coffee Preference: **{coffee}**
- 💰 Budget: **{budget}**
- 🚶 Walkability: **{walkability}**
- ⭐ Minimum Rating: **{rating}**

Based on patterns learned from the training data,
**{prediction}** was determined to be the best match.
"""
        )

        # -----------------------------
        # Google Maps
        # -----------------------------
        google_link = (
            "https://www.google.com/maps/search/"
            + urllib.parse.quote(str(prediction))
        )

        st.link_button(
            "📍 View on Google Maps",
            google_link
        )

        # -----------------------------
        # Shop Details
        # -----------------------------
        if not shop_info.empty:

            info = shop_info.iloc[0]

            st.markdown("---")

            st.header(f"☕ {info['shop']}")

            # Rating
            if "rating" in shops.columns:

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "⭐ Rating",
                        f"{info['rating']}"
                    )

                # Price
                if "price" in shops.columns:
                    with col2:
                        st.metric(
                            "💰 Price",
                            f"{info['price']}"
                        )

                elif "budget" in shops.columns:
                    with col2:
                        st.metric(
                            "💰 Price",
                            f"{info['budget']}"
                        )

            # Review
            if "review" in shops.columns:

                st.markdown("### Why you'll like it")

                st.write(info["review"])

        else:

            st.warning(
                "The model predicted this coffee shop, but additional "
                "information for the shop could not be found in shops.csv."
            )

        # -----------------------------
        # Model Information
        # -----------------------------
        st.markdown("---")

        st.info(
            "Recommendation generated using a Decision Tree "
            "Machine Learning model."
        )

    except Exception as e:

        st.error(
            "Something went wrong while generating the recommendation."
        )

        st.write(
            "Please make sure your model, encoders, and shops.csv "
            "file are correctly configured."
        )

        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    """
CoffeeRun AI

Graduate Machine Learning Project

Built using:

• Python

• Streamlit

• scikit-learn

• Decision Tree Classifier

Prototype dataset created for educational purposes.
"""
)
