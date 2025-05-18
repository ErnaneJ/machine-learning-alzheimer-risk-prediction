import streamlit as st
import numpy as np
import pandas as pd
import torch

def get_index_from_options(options, value):
    return next(i for i, (val, _) in enumerate(options) if val == value)

def show_prediction_page(model, scaler, feature_names):
    st.title("🧠 Predict Alzheimer Risk")
    st.markdown("Fill out the form with patient data to predict the risk of Alzheimer's disease.")

    with st.expander("ℹ️ About the Input Parameters"):
        st.markdown("This section describes the meaning and clinical relevance of each input feature:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            ### 🧑 Demographic & Lifestyle
            - **Age**: Age of the patient in years (60–90).
            - **Gender**: Biological sex of the patient (Male or Female).
            - **Ethnicity**: Ethnic background, which can affect health risks.
            - **Education Level**: Higher education is associated with cognitive reserve.
            - **BMI**: Body Mass Index, used to assess healthy body weight.
            - **Smoking**: Smoking increases vascular and neurological risks.
            - **Alcohol Consumption**: Moderate or high intake may affect cognitive health.
            - **Physical Activity**: Protective factor for cognitive and cardiovascular health.
            - **Diet Quality**: Higher scores indicate healthier dietary patterns.
            - **Sleep Quality**: Poor sleep is linked to cognitive impairment.
            """)
        with col2:
            st.markdown("""
            ### 🩺 Health & Medical Info
            - **Family History of Alzheimer's**: Genetic predisposition to the disease.
            - **Cardiovascular Disease**: Increases risk of dementia.
            - **Diabetes**: Associated with increased risk of Alzheimer's.
            - **Depression**: Can be a symptom and a risk factor.
            - **Head Injury**: Traumatic brain injuries can lead to cognitive decline.
            - **Hypertension**: High blood pressure affects brain health.
            - **Systolic/Diastolic BP**: Blood pressure values in mmHg.
            - **Cholesterol (Total, LDL, HDL, Triglycerides)**: Poor lipid profiles are linked to vascular dementia.
            - **MMSE**: Mini-Mental State Examination score (0–30), measures cognitive function.
            - **Functional Assessment**: Ability to perform daily living tasks.
            - **ADL**: Activities of Daily Living score, reflects independence.
            """)
        with col3:
            st.markdown("""
            ### 🧠 Cognitive & Behavioral Symptoms
            - **Memory Complaints**: Subjective cognitive decline.
            - **Behavioral Problems**: Issues like agitation, aggression.
            - **Confusion / Disorientation**: Disruption of mental clarity or awareness.
            - **Personality Changes**: Alterations in mood or social behavior.
            - **Difficulty Completing Tasks**: Reflects executive dysfunction.
            - **Forgetfulness**: Common symptom of cognitive decline.
            """)

        st.markdown("These features are used collectively by the machine learning model to assess the **risk of Alzheimer's disease** based on patterns learned from the dataset.")

    _, _, col0, col01 = st.columns(4)
    with col0:
        if st.button("🔵 Load LOW-RISK Example", use_container_width=True):
            st.session_state.update({
                "Age": 65,
                "Gender": 0,
                "Ethnicity": 0,
                "EducationLevel": 3,
                "BMI": 24.0,
                "Smoking": 0,
                "AlcoholConsumption": 2.0,
                "PhysicalActivity": 6.0,
                "DietQuality": 9.0,
                "SleepQuality": 8.0,
                "FamilyHistoryAlzheimers": 0,
                "CardiovascularDisease": 0,
                "Diabetes": 0,
                "Depression": 0,
                "HeadInjury": 0,
                "Hypertension": 0,
                "SystolicBP": 115,
                "DiastolicBP": 75,
                "CholesterolTotal": 180,
                "CholesterolLDL": 90,
                "CholesterolHDL": 60,
                "CholesterolTriglycerides": 130,
                "MMSE": 29,
                "FunctionalAssessment": 9,
                "MemoryComplaints": 0,
                "BehavioralProblems": 0,
                "ADL": 9,
                "Confusion": 0,
                "Disorientation": 0,
                "PersonalityChanges": 0,
                "DifficultyCompletingTasks": 0,
                "Forgetfulness": 0,
            })
            st.toast("Low-risk example loaded successfully!", icon="✅")
    with col01:
            if st.button("🔴 Load HIGH-RISK Example", use_container_width=True):
                st.session_state.update({
                    "Age": 88,
                    "Gender": 1,
                    "Ethnicity": 1,
                    "EducationLevel": 0,
                    "BMI": 30.0,
                    "Smoking": 1,
                    "AlcoholConsumption": 12.0,
                    "PhysicalActivity": 0.0,
                    "DietQuality": 3.0,
                    "SleepQuality": 4.0,
                    "FamilyHistoryAlzheimers": 1,
                    "CardiovascularDisease": 1,
                    "Diabetes": 1,
                    "Depression": 1,
                    "HeadInjury": 1,
                    "Hypertension": 1,
                    "SystolicBP": 160,
                    "DiastolicBP": 100,
                    "CholesterolTotal": 280,
                    "CholesterolLDL": 190,
                    "CholesterolHDL": 30,
                    "CholesterolTriglycerides": 320,
                    "MMSE": 9,
                    "FunctionalAssessment": 3,
                    "MemoryComplaints": 1,
                    "BehavioralProblems": 1,
                    "ADL": 2,
                    "Confusion": 1,
                    "Disorientation": 1,
                    "PersonalityChanges": 1,
                    "DifficultyCompletingTasks": 1,
                    "Forgetfulness": 1,
                })
                st.toast("High-risk example loaded successfully!", icon="✅")
       
    with st.container(border=True):
        st.markdown("#### 🧑 Demographic & Lifestyle")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", 60, 90, value=st.session_state.get("Age", 75))
            bmi = st.number_input("BMI", 15.0, 40.0, value=float(st.session_state.get("BMI", 25.0)))
            sleep_quality = st.number_input("Sleep Quality (0–10)", 0.0, 10.0, value=float(st.session_state.get("SleepQuality", 6.0)))
            systolic_bp = st.number_input("Systolic BP", 90, 180, value=st.session_state.get("SystolicBP", 120))
        with col2:
            gender_opts = [(0, "Male"), (1, "Female")]
            gender = st.selectbox("Gender", gender_opts, index=get_index_from_options(gender_opts, st.session_state.get("Gender", 0)), format_func=lambda x: x[1])[0]
            alcohol = st.number_input("Alcohol Consumption", 0.0, 20.0, value=float(st.session_state.get("AlcoholConsumption", 5.0)))
            family_opts = [(0, "No"), (1, "Yes")]
            family_history = st.selectbox("Family History", family_opts, index=get_index_from_options(family_opts, st.session_state.get("FamilyHistoryAlzheimers", 0)), format_func=lambda x: x[1])[0]
            diastolic_bp = st.number_input("Diastolic BP", 60, 120, value=st.session_state.get("DiastolicBP", 80))
        with col3:
            ethnicity_opts = [(0, "Caucasian"), (1, "African-American"), (2, "Asian"), (3, "Other")]
            ethnicity = st.selectbox("Ethnicity", ethnicity_opts, index=get_index_from_options(ethnicity_opts, st.session_state.get("Ethnicity", 0)), format_func=lambda x: x[1])[0]
            education_opts = [(0, "None"), (1, "High School"), (2, "Bachelor"), (3, "Higher")]
            education = st.selectbox("Education Level", education_opts, index=get_index_from_options(education_opts, st.session_state.get("EducationLevel", 1)), format_func=lambda x: x[1])[0]
            smoking_opts = [(0, "No"), (1, "Yes")]
            smoking = st.selectbox("Smoker", smoking_opts, index=get_index_from_options(smoking_opts, st.session_state.get("Smoking", 0)), format_func=lambda x: x[1])[0]
            physical_activity = st.number_input("Physical Activity", 0.0, 10.0, value=float(st.session_state.get("PhysicalActivity", 3.0)))

        st.markdown("#### 🩺 Health & Medical Info")
        col4, col5, col6 = st.columns(3)
        with col4:
            diet_quality = st.number_input("Diet Quality", 0.0, 10.0, value=float(st.session_state.get("DietQuality", 5.0)))
            cholesterol_total = st.number_input("Cholesterol Total", 150.0, 300.0, value=float(st.session_state.get("CholesterolTotal", 200.0)))
            cholesterol_hdl = st.number_input("HDL Cholesterol", 20.0, 100.0, value=float(st.session_state.get("CholesterolHDL", 50.0)))
            mmse = st.number_input("MMSE", 0.0, 30.0, value=float(st.session_state.get("MMSE", 26.0)))
        with col5:
            cholesterol_ldl = st.number_input("LDL Cholesterol", 50.0, 200.0, value=float(st.session_state.get("CholesterolLDL", 100.0)))
            cholesterol_triglycerides = st.number_input("Triglycerides", 50.0, 400.0, value=float(st.session_state.get("CholesterolTriglycerides", 150.0)))
            functional_assessment = st.number_input("Functional Assessment", 0.0, 10.0, value=float(st.session_state.get("FunctionalAssessment", 8.0)))
            adl = st.number_input("ADL", 0.0, 10.0, value=float(st.session_state.get("ADL", 8.0)))
        with col6:
            cardiovascular = st.selectbox("Cardiovascular Disease", family_opts, index=get_index_from_options(family_opts, st.session_state.get("CardiovascularDisease", 0)), format_func=lambda x: x[1])[0]
            diabetes = st.selectbox("Diabetes", family_opts, index=get_index_from_options(family_opts, st.session_state.get("Diabetes", 0)), format_func=lambda x: x[1])[0]
            depression = st.selectbox("Depression", family_opts, index=get_index_from_options(family_opts, st.session_state.get("Depression", 0)), format_func=lambda x: x[1])[0]
            head_injury = st.selectbox("Head Injury", family_opts, index=get_index_from_options(family_opts, st.session_state.get("HeadInjury", 0)), format_func=lambda x: x[1])[0]
        hypertension = st.selectbox("Hypertension", family_opts, index=get_index_from_options(family_opts, st.session_state.get("Hypertension", 0)), format_func=lambda x: x[1])[0]

        st.markdown("#### 🧠 Cognitive & Behavioral Symptoms")
        col7, col8, col9 = st.columns(3)
        with col7:
            memory_complaints = st.selectbox("Memory Complaints", family_opts, index=get_index_from_options(family_opts, st.session_state.get("MemoryComplaints", 0)), format_func=lambda x: x[1])[0]
            behavioral_problems = st.selectbox("Behavioral Problems", family_opts, index=get_index_from_options(family_opts, st.session_state.get("BehavioralProblems", 0)), format_func=lambda x: x[1])[0]
        with col8:
            confusion = st.selectbox("Confusion", family_opts, index=get_index_from_options(family_opts, st.session_state.get("Confusion", 0)), format_func=lambda x: x[1])[0]
            disorientation = st.selectbox("Disorientation", family_opts, index=get_index_from_options(family_opts, st.session_state.get("Disorientation", 0)), format_func=lambda x: x[1])[0]
        with col9:
            personality_changes = st.selectbox("Personality Changes", family_opts, index=get_index_from_options(family_opts, st.session_state.get("PersonalityChanges", 0)), format_func=lambda x: x[1])[0]
            difficulty_tasks = st.selectbox("Difficulty Completing Tasks", family_opts, index=get_index_from_options(family_opts, st.session_state.get("DifficultyCompletingTasks", 0)), format_func=lambda x: x[1])[0]
        
        forgetfulness = st.selectbox("Forgetfulness", family_opts, index=get_index_from_options(family_opts, st.session_state.get("Forgetfulness", 0)), format_func=lambda x: x[1])[0]
        
        submit_button = st.button("🎯 Predict", use_container_width=True)

    if submit_button:
        input_data = {
            "Age": age,
            "Gender": gender,
            "Ethnicity": ethnicity,
            "EducationLevel": education,
            "BMI": bmi,
            "Smoking": smoking,
            "AlcoholConsumption": alcohol,
            "PhysicalActivity": physical_activity,
            "DietQuality": diet_quality,
            "SleepQuality": sleep_quality,
            "FamilyHistoryAlzheimers": family_history,
            "CardiovascularDisease": cardiovascular,
            "Diabetes": diabetes,
            "Depression": depression,
            "HeadInjury": head_injury,
            "Hypertension": hypertension,
            "SystolicBP": systolic_bp,
            "DiastolicBP": diastolic_bp,
            "CholesterolTotal": cholesterol_total,
            "CholesterolLDL": cholesterol_ldl,
            "CholesterolHDL": cholesterol_hdl,
            "CholesterolTriglycerides": cholesterol_triglycerides,
            "MMSE": mmse,
            "FunctionalAssessment": functional_assessment,
            "MemoryComplaints": memory_complaints,
            "BehavioralProblems": behavioral_problems,
            "ADL": adl,
            "Confusion": confusion,
            "Disorientation": disorientation,
            "PersonalityChanges": personality_changes,
            "DifficultyCompletingTasks": difficulty_tasks,
            "Forgetfulness": forgetfulness,
            "Diagnosis": 0
        }

        input_df = pd.DataFrame([input_data])[feature_names]
        input_scaled = scaler.transform(input_df)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        output = model(input_tensor)
        prediction = (output.item() >= 0.5)

        st.success(f"The predicted risk of Alzheimer's is: {'Positive' if prediction else 'Negative'} ({output.item():.2f})")