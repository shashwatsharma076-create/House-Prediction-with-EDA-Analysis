import streamlit as st
import pandas as pd
import numpy as np
from house_price_prediction import load_data, preprocess_data, train_model, evaluate_model

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Predict house prices using Linear Regression")

st.sidebar.header("Input Features")

def user_input():
    sq_ft = st.sidebar.slider("Square Footage", 1000, 4000, 2000)
    bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3])
    year_built = st.sidebar.slider("Year Built", 1950, 2024, 2000)
    lot_size = st.sidebar.slider("Lot Size (acres)", 0.1, 2.0, 0.5)
    garage = st.sidebar.selectbox("Garage Spots", [0, 1, 2, 3])
    
    return pd.DataFrame({
        'square_footage': [sq_ft],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'year_built': [year_built],
        'lot_size': [lot_size],
        'garage_spots': [garage]
    })

df = load_data()
X, y = preprocess_data(df)
model, X_train, X_test, y_train, y_test = train_model(X, y)

input_df = user_input()

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Price: ${prediction:,.2f}")
    
    st.subheader("Price Breakdown")
    st.write(f"Base price for {input_df['square_footage'].values[0]} sq ft: ${input_df['square_footage'].values[0] * 200:,}")
    st.write(f"Bedrooms (+{input_df['bedrooms'].values[0] * 10000:,}): ${input_df['bedrooms'].values[0] * 10000:,}")
    st.write(f"Bathrooms (+{input_df['bathrooms'].values[0] * 15000:,}): ${input_df['bathrooms'].values[0] * 15000:,}")

st.subheader("📊 Model Performance")
y_pred = model.predict(X_test)
rmse = np.sqrt(((y_test - y_pred)**2).mean())
r2 = 1 - (((y_test - y_pred)**2).sum() / ((y_test - y_test.mean())**2).sum())
st.metric("RMSE", f"${rmse:,.0f}")
st.metric("R² Score", f"{r2:.3f}")

st.subheader("📈 Feature Importance")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
pd.Series(model.coef_, index=X.columns).plot(kind='bar', ax=ax, color='teal')
st.pyplot(fig)