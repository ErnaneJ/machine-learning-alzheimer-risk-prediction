
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def show_dashboard(model, scaler, feature_names):
    st.title("🧠 Alzheimer Risk Prediction - Project Overview")

    st.markdown("""
    This Streamlit application showcases a complete machine learning pipeline to predict Alzheimer's disease risk.

    The application is powered by a logistic regression model implemented in PyTorch, trained on a synthetic dataset of patients with various demographic, health, and behavioral indicators. Below, we describe every step of the development process and results.

    ## 📁 Dataset
    The dataset includes records of patients aged 60–90, with features related to:
    - Demographics (Age, Gender, Ethnicity, Education)
    - Lifestyle (Smoking, Alcohol, Physical Activity, Diet, Sleep)
    - Medical history (Cardiovascular, Diabetes, Depression, etc.)
    - Cognitive assessments (MMSE, ADL, etc.)
    - Target: **Diagnosis** (0 = No Alzheimer's, 1 = Alzheimer's)

    ## 🔍 Exploratory Data Analysis (EDA)
    Before modeling, we conducted an analysis to understand:
    - Data distribution
    - Presence of outliers
    - Class imbalance
    - Correlation between variables

    ### 📌 Class Balance
    """)

    df = pd.read_csv("models/sample_data.csv")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Distribution of Target Variable (Diagnosis)")
    fig = px.histogram(df, x="Diagnosis", color="Diagnosis", nbins=2, text_auto=True, title="Diagnosis Distribution")
    st.plotly_chart(fig)

    st.markdown("""
    The dataset is slightly imbalanced, with a majority of class 0 (non-Alzheimer's), but still suitable for classification.
    """)

    st.subheader("Histogram of Numeric Features")
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col not in ['PatientID', 'Diagnosis']]
    n_cols = 5
    n_rows = -(-len(numeric_cols) // n_cols)
    fig = make_subplots(rows=n_rows, cols=n_cols,
                    subplot_titles=numeric_cols,
                    horizontal_spacing=0.05,
                    vertical_spacing=0.1)
    for i, col in enumerate(numeric_cols):
        row = i // n_cols + 1
        col_num = i % n_cols + 1

        fig.add_trace(
            go.Histogram(
                x=df[col],
                nbinsx=20,
                marker_color='lightsteelblue',
                marker_line_color='black',
                marker_line_width=1
            ),
            row=row, col=col_num
        )
    
    fig.update_layout(
        height=n_rows * 300,
        width=n_cols * 300,
        title_text="Histograms of Numeric Features",
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 📊 Correlation Analysis
    We evaluated the correlation between all numerical features and the target variable:
    """)
    corr = df[feature_names + ['Diagnosis']].corr()
    corr_fig = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix")
    st.plotly_chart(corr_fig)
    st.markdown("""
    Correlation matrices helped identify which features might be redundant or not useful.

    ## 🧹 Feature Engineering and Data Cleaning
    - Removed columns with high missing rates or low predictive power (e.g., `PatientID`, `DoctorInCharge`)
    - Applied Min-Max scaling using `sklearn.preprocessing`
    - Converted all categorical values to numerical format
    - Ensured compatibility with PyTorch input format (tensors)

    ## 🧠 Model Implementation (PyTorch)
    We implemented logistic regression from scratch using PyTorch, including:
    - Forward pass
    - Sigmoid activation
    - Binary cross entropy loss

    The class was based on the base structure provided by the professor.

    ## 🏋️ Training the Model
    - Split: 80% training / 20% test
    - Optimizer: SGD
    - Epochs: 100
    - Loss function: Binary Cross Entropy
    """)

    st.subheader("Training Loss over Epochs")
    st.image("assets/training-loss-over-epochs.png", caption="Training Loss Curve")

    st.markdown("""
    Example loss progression:
    ```
    Epoch [1/100], Loss: 0.7240
    Epoch [10/100], Loss: 0.4885
    Epoch [20/100], Loss: 0.4127
    ...
    Epoch [90/100], Loss: 0.3604
    Epoch [100/100], Loss: 0.3598
    ```
    As we can see, the model converged well over 100 epochs.

    ## 📈 Model Evaluation

    After training, the model was evaluated on the test set. Below are the results:

    - **Accuracy**: 81.63%
    - **Precision**: 74.17%
    - **Recall**: 73.68%
    - **F1 Score**: 73.93%

    These metrics indicate a good balance between precision and recall, with the model capable of generalizing well to unseen data.

    ### 🔀 Confusion Matrix
    """)
    st.image("assets/confusion-matrix.png", caption="Confusion Matrix")

    st.markdown("""
    The confusion matrix confirms a solid performance with relatively few false positives or negatives.

    ## 💾 Model Export
    The final model and necessary preprocessing components were saved:
    - `logistic_model.pkl`: Trained PyTorch model wrapped for inference
    - `logistic_model_weights.pth`: Model weights
    - `features.pkl`: List of selected input features for inference

    These are used at runtime to make predictions in the Streamlit app.
    """)
