import streamlit as st
from utils.model_loader import load_model
from pages.main_dashboard import show_dashboard
from pages.predict_form import show_prediction_page

st.set_page_config(page_title="Alzheimer Risk Prediction", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Project Overview", "🧠 Alzheimer Risk Prediction"])

model, scaler, feature_names = load_model()

if page == "🏠 Project Overview":
    show_dashboard(model, scaler, feature_names)
elif page == "🧠 Alzheimer Risk Prediction":
    show_prediction_page(model, scaler, feature_names)