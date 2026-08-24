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
        # The model already returns the coffee shop name.
        prediction = model.predict(user_input)[0]

        # Make sure prediction is a normal string
        prediction = str(prediction)

        # -----------------------------
        # Match Score
        # -----------------------------
        probabilities = model.predict_proba(user_input)[0]
        match = round(max(probabilities) * 100)

        # -----------------------------
        # Find Shop Information
        # -----------------------------
        if "shop" not in shops.columns:
            st.error(
                "Your shops.csv file must contain a column named 'shop'."
            )
            st.stop()

        shop_info = shops[
            shops["shop"].astype(str).str.strip().str.lower()
            == prediction.strip().lower()
        ]

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
**{prediction}** was determined to be the best overall match.
"""
        )

        # -----------------------------
        # Google Maps
        # -----------------------------
        google_link = (
            "https://www.google.com/maps/search/"
            + urllib.parse.quote(prediction)
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

            col1, col2 = st.columns(2)

            with col1:
                if "rating" in shops.columns:
                    st.metric(
                        "⭐ Rating",
                        info["rating"]
                    )

            with col2:
                if "price" in shops.columns:
                    st.metric(
                        "💰 Price",
                        info["price"]
                    )

                elif "budget" in shops.columns:
                    st.metric(
                        "💰 Price",
                        info["budget"]
                    )

            if "review" in shops.columns:

                st.markdown("### Why you'll like it")

                st.write(info["review"])

        else:

            st.warning(
                f"We couldn't find additional information for "
                f"**{prediction}** in shops.csv."
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
