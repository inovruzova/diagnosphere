import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
from bayes_net import BayesNet, MultiClassBayesNode, enumeration_ask, compute_cpt

# Load Mental Health and COVID-19 data
final_mental_data = pd.read_csv('../data/final_mental_data.csv')
covid_data = pd.read_csv('../data/covid_preprocessed.csv')

# Set up the page layout and style
st.set_page_config(page_title="🩺Diagnosphere", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f7f9fc;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #5aa9e6, #004e89);
            color: white;
            border: none;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #81c3e6, #004e89);
        }
        .stButton>button:active {
            background: linear-gradient(90deg, #004e89, #81c3e6);
        }
        h1, h2 {
            color: #004e89;
        }
        .sidebar {
            background-color: #e3f2fd;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "home"

# Sidebar Navigation Panel
st.sidebar.header("Navigation")
st.sidebar.button("Go Back to Home", on_click=lambda: setattr(st.session_state, "page", "home"))

# App title
st.title("🩺Diagnosphere: Get Clinical Insights!")

# Home Page
if st.session_state.page == "home":
    st.subheader("Choose an area to explore clinical insights.")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Mental Health"):
            st.session_state.page = "mental_health"
    with col2:
        if st.button("COVID-19"):
            st.session_state.page = "covid_19"
    with col3:
        if st.button("Breast Cancer"):
            st.session_state.page = "breast_cancer"

# Mental Health Page
elif st.session_state.page == "mental_health":
    st.header("Mental Health Assessment")
    st.write("Answer the following questions to get insights into your mental health.")

    # Load mappings
    with open('../Preprocessing/mental_label_mappings.json', 'r') as f:
        mappings = json.load(f)
    with open('../Preprocessing/mental_work_hours_bins.json', 'r') as f:
        bin_edges = json.load(f)

    # Collect user input
    age_raw = st.number_input("Enter your age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Choose your gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
    occupation = st.selectbox("Which describes your occupation area the best?", ["Other", "Healthcare", "Engineering", "Finance", "Sales", "Education", "IT"])
    country = st.selectbox("Choose the country you are currently living in", ["Australia", "India", "USA", "Germany", "UK", "Canada", "Other"])
    consultation_history = st.selectbox("Do you have a consultation history?", ["Yes", "No"])
    sleep_hours_raw = st.number_input("How many hours of sleep do you get per night?", min_value=0, max_value=24, step=1)
    work_hours_raw = st.number_input("How many hours do you work on average per week?", min_value=0, max_value=168, step=1)
    physical_activity_raw = st.number_input("How many hours do you spend on physical activities per day?", min_value=0, max_value=24, step=1)

    # Discretize continuous variables
    age = "Young" if age_raw <= 30 else "Middle-aged" if age_raw <= 66 else "Senior"
    sleep_hours = "Low" if sleep_hours_raw <= 6 else "Medium" if sleep_hours_raw <= 8 else "High"
    physical_activity = "Low" if physical_activity_raw <= 3 else "Medium" if physical_activity_raw <= 7 else "High"

    if work_hours_raw <= bin_edges[1]:
        work_hours = 'Low'
    elif work_hours_raw <= bin_edges[2]:
        work_hours = 'Medium'
    else:
        work_hours = 'High'

    for col in final_mental_data.columns:
        final_mental_data[col] = final_mental_data[col].astype('category')

    # Map user input to encoded values
    age_encoded = mappings["Age"][age]
    gender_encoded = mappings["Gender"][gender]
    occupation_encoded = mappings["Occupation"][occupation]
    country_encoded = mappings["Country"][country]
    consultation_history_encoded = mappings["Consultation_History"][consultation_history]
    sleep_hours_encoded = mappings["Sleep_Hours"][sleep_hours]
    work_hours_encoded = mappings["Work_Hours"][work_hours]
    physical_activity_encoded = mappings["Physical_Activity_Hours"][physical_activity]

    # Prepare the input data
    evidence = {
        "Age": age_encoded,
        "Gender": gender_encoded,
        "Occupation": occupation_encoded,
        "Country": country_encoded,
        "Consultation_History": consultation_history_encoded,
        "Sleep_Hours": sleep_hours_encoded,
        "Work_Hours": work_hours_encoded,
        "Physical_Activity_Hours": physical_activity_encoded
    }

    # Define Bayesian Network structure
    cpt_mental_health = compute_cpt(final_mental_data, "Mental_Health_Condition", ["Physical_Activity_Hours", "Gender"])
    mental_health_node = MultiClassBayesNode("Mental_Health_Condition", ["Physical_Activity_Hours", "Gender"], cpt_mental_health)

    mental_health_bn = BayesNet([mental_health_node])

    if st.button("Get Result"):
        try:
            # Predict the target variable using the Bayesian Network
            result = enumeration_ask("Mental_Health_Condition", evidence, mental_health_bn)
            prediction = max(result.prob, key=result.prob.get)  # Most likely value
            prediction_text = "Yes" if prediction == 1 else "No"
            st.success(f"Your mental health condition result: {prediction_text}")

            if prediction == 1:
                st.info("We recommend consulting a healthcare professional. Here are some tips: Maintain a healthy routine, stay active, and seek support from friends and family.")
            else:
                st.info("You seem to be in good mental health! Keep maintaining a healthy lifestyle and stay positive.")
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")

# COVID-19 Page
elif st.session_state.page == "covid_19":
    st.header("COVID-19 Symptom Checker")
    st.write("Answer the following questions to check your COVID-19 classification risk.")

    # Collect user input
    age_group = st.selectbox("Select your age group", ['<20', '20-40', '40-60', '60-80', '80+'])
    sex = st.selectbox("Select your sex", ["Female", "Male"])
    obesity = st.selectbox("Are you obese?", ["Yes", "No"])
    diabetes = st.selectbox("Do you have diabetes?", ["Yes", "No"])
    pneumonia = st.selectbox("Do you have pneumonia?", ["Yes", "No"])
    icu = st.selectbox("Have you been admitted to ICU (Intensive Care Unit)?", ["Yes", "No"])

    # Encode inputs
    sex_encoded = 1 if sex == "Female" else 2
    obesity_encoded = 1 if obesity == "Yes" else 2
    diabetes_encoded = 1 if diabetes == "Yes" else 2
    pneumonia_encoded = 1 if pneumonia == "Yes" else 2
    icu_encoded = 1 if icu == "Yes" else 2

    # Build evidence
    evidence = {
        "AGE_GROUP": age_group,
        "SEX": sex_encoded,
        "OBESITY": obesity_encoded,
        "DIABETES": diabetes_encoded,
        "PNEUMONIA": pneumonia_encoded,
        "ICU": icu_encoded
    }

    # Define Bayesian Network
    cpt_classification = compute_cpt(covid_data, 'CLASIFFICATION_FINAL', ['SEX', 'OBESITY', 'DIABETES', 'ICU'])
    classification_node = MultiClassBayesNode("CLASIFFICATION_FINAL", ['SEX', 'OBESITY', 'DIABETES', 'ICU'], cpt_classification)
    covid_bn = BayesNet([classification_node])

    if st.button("Check COVID-19 Risk"):
        try:
            result = enumeration_ask("CLASIFFICATION_FINAL", evidence, covid_bn)
            prediction = max(result.prob, key=result.prob.get)
            prediction_text = "Positive" if prediction == 1 else "Negative"
            st.success(f"Your COVID-19 classification result: {prediction_text}")
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")

# Breast Cancer Page
elif st.session_state.page == "breast_cancer":
    st.header("Breast Cancer Risk Assessment")
    st.write("Feature under development. Please consult a doctor for accurate diagnosis.")

# Footer
st.sidebar.write("Navigate through the app to explore features.")