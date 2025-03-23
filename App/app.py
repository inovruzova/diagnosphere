import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json
import ast
from bayes_net import BayesNet, MultiClassBayesNode, enumeration_ask

# Load Mental Health and COVID-19 data
final_mental_data = pd.read_csv('../data/final_mental_data.csv')
covid_data = pd.read_csv('../data/covid_preprocessed.csv')

st.set_page_config(page_title="🩺Diagnosphere", layout="wide", initial_sidebar_state="expanded")

# Convert CPTs back to proper structure
def json_to_cpts(cpt_json):
    return {
        var: {
            eval(parent_comb): {eval(k): v for k, v in target_probs.items()}
            for parent_comb, target_probs in cpt.items()
        }
        for var, cpt in cpt_json.items()
    }

# Custom CSS
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

    # Load mappings and precomputed CPTs
    try:
        with open('../Preprocessing/mental_label_mappings.json', 'r') as f:
            mappings = json.load(f)
        with open('../Preprocessing/mental_work_hours_bins.json', 'r') as f:
            bin_edges = json.load(f)
        with open('../Bayesian From Scratch/mental_cpts.json', 'r') as f:
            mental_cpts = json.load(f)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")

    cpts = json_to_cpts(mental_cpts)

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

    try:
        age_encoded = mappings["Age"][age]
        gender_encoded = mappings["Gender"][gender]
        occupation_encoded = mappings["Occupation"][occupation]
        country_encoded = mappings["Country"][country]
        consultation_history_encoded = mappings["Consultation_History"][consultation_history]
        sleep_hours_encoded = mappings["Sleep_Hours"][sleep_hours]
        work_hours_encoded = mappings["Work_Hours"][work_hours]
        physical_activity_encoded = mappings["Physical_Activity_Hours"][physical_activity]
    except KeyError as e:
        st.error(f"Mapping error: {e}")

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
    try:
        mental_health_node = MultiClassBayesNode("Mental_Health_Condition", ["Physical_Activity_Hours", "Gender"], cpts["cpt_mental_health"])
        activity_node = MultiClassBayesNode("Physical_Activity_Hours", ["Country"], cpts["cpt_activity"])
        gender_node = MultiClassBayesNode("Gender", ["Occupation"], cpts["cpt_gender"])
        country_node = MultiClassBayesNode("Country", ["Age"], cpts["cpt_country"])
        age_node = MultiClassBayesNode("Age", [], cpts["cpt_age"])
        occupation_node = MultiClassBayesNode("Occupation", [], cpts["cpt_occupation"])

        mental_health_bn = BayesNet([
            age_node,
            occupation_node,
            gender_node,
            country_node,
            activity_node,
            mental_health_node
        ])
    except KeyError as e:
        st.error(f"CPT key error: {e}")

    if st.button("Get Result"):
        try:
            result = enumeration_ask("Mental_Health_Condition", evidence, mental_health_bn)
            prediction = max(result.prob, key=result.prob.get)
            prediction_text = "Yes" if prediction == 1 else "No"
            st.success(f"Your mental health condition result: {prediction_text}")
            
            # Display probabilities
            st.write("Probabilities:")
            st.write(f"Yes (1): {result.prob.get(1,0):.4f}")
            st.write(f"No (0): {result.prob.get(0,0):.4f}")

            if prediction == 1:
                st.info("We recommend consulting a healthcare professional. Maintain a healthy routine, stay active, and seek support.")
            else:
                st.info("You seem to be in good mental health! Keep maintaining a healthy lifestyle and stay positive.")
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")

# COVID-19 Page
elif st.session_state.page == "covid_19":
    st.header("COVID-19 Symptom Checker")
    st.write("Answer the following questions to get insights into COVID-19 classification.")

    try:
        with open("../Bayesian From Scratch/covid_cpts.json", "r") as f:
            covid_cpts = json.load(f)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")

    cpts = json_to_cpts(covid_cpts)

    # Define Bayesian Network
    try:
        classification_node = MultiClassBayesNode("CLASIFFICATION_FINAL", ['SEX', 'OBESITY', 'DIABETES', 'ICU'], cpts['cpt_classification'])
        pneumonia_node = MultiClassBayesNode("PNEUMONIA", ["AGE_GROUP"], cpts['cpt_pneumonia'])
        icu_node = MultiClassBayesNode("ICU", ["PNEUMONIA"], cpts['cpt_icu'])
        age_node = MultiClassBayesNode("AGE_GROUP", [], cpts['cpt_age_group'])
        obesity_node = MultiClassBayesNode("OBESITY", [], cpts['cpt_obesity_group'])
        sex_node = MultiClassBayesNode("SEX", [], cpts['cpt_sex_group'])
        diabetes_node = MultiClassBayesNode("DIABETES", [], cpts['cpt_diabetes_group'])

        covid_bn = BayesNet([
            age_node,
            sex_node,
            obesity_node,
            diabetes_node,
            pneumonia_node,
            icu_node,
            classification_node
        ])
    except KeyError as e:
        st.error(f"CPT key error: {e}")

    st.write("### Please provide the following details:")
    age_group = st.selectbox("Age Group", ["<20", "20-40", "40-60", "60-80", "80+"])
    sex = st.selectbox("Sex", ["Female", "Male"])
    obesity = st.selectbox("Do you have obesity?", ["Yes", "No"])
    diabetes = st.selectbox("Do you have diabetes?", ["Yes", "No"])
    pneumonia = st.selectbox("Do you have pneumonia?", ["Yes", "No"])
    icu = st.selectbox("Have you been admitted to ICU?", ["Yes", "No"])

    if (age_group == "<20"):
        age_evidence = 0
    elif (age_group == "20-40"):
        age_evidence = 1
    elif (age_group == "40-60"):
        age_evidence = 2
    elif (age_group == "60-80"):
        age_evidence = 3
    elif (age_group == "80+"):
        age_evidence = 4

    try:
        evidence = {
            "AGE_GROUP": age_evidence,
            "SEX": 1 if sex == "Female" else 2,
            "OBESITY": 1 if obesity == "Yes" else 2,
            "DIABETES": 1 if diabetes == "Yes" else 2,
            "PNEUMONIA": 1 if pneumonia == "Yes" else 2,
            "ICU": 1 if icu == "Yes" else 2,
        }
    except KeyError as e:
        st.error(f"Mapping error: {e}")

    if st.button("Get COVID-19 Classification"):
        try:
            result = enumeration_ask("CLASIFFICATION_FINAL", evidence, covid_bn)
            prediction = max(result.prob, key=result.prob.get)
            prediction_text = "Positive" if prediction == 1 else "Negative"
            st.success(f"Your COVID-19 classification result: {prediction_text}")
            
            # Display probabilities
            st.write("Probabilities:")
            st.write(f"Positive (1): {result.prob.get(1,0):.4f}")
            st.write(f"Negative (2): {result.prob.get(0,0):.4f}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Breast Cancer Page
elif st.session_state.page == "breast_cancer":
    st.header("Breast Cancer Risk Assessment")
    st.write("Please enter the clinical measurements. We'll discretize them internally as done during preprocessing.")

    # Meanings of each feature as previously provided:
    st.markdown("""
    **Feature Meanings:**
    - **radius_mean**: mean of distances from center to points on the perimeter
    - **texture_mean**: standard deviation of gray-scale values
    - **perimeter_mean**: mean size of the core tumor
    - **area_mean**: mean area of the tumor
    - **smoothness_mean**: mean of local variation in radius lengths
    - **compactness_mean**: mean of perimeter^2 / area - 1.0
    - **concavity_mean**: mean of severity of concave portions of the contour
    - **concave points_mean**: mean for number of concave portions of the contour
    - **symmetry_mean**: measure of symmetry of the tumor cells
    - **fractal_dimension_mean**: mean for "coastline approximation" (fractality) of the tumor
    """)

    # Load Precomputed CPTs for Breast Cancer
    try:
        with open("../Bayesian From Scratch/cancer_cpts.json", "r") as f:
            cancer_cpts = json.load(f)
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        st.stop()

    cpts = json_to_cpts(cancer_cpts)

    # Load bin edges
    try:
        with open("../Preprocessing/cancer_bins.json", "r") as f:
            cancer_bins = json.load(f)
    except FileNotFoundError as e:
        st.error("Bin edges file not found. Please ensure cancer_bins.json is in the correct directory.")
        st.stop()
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        st.stop()

    # Define the Bayesian Network structure for breast cancer
    try:
        diagnosis_node = MultiClassBayesNode("diagnosis", [], cpts["cpt_diagnosis"])
        concave_points_node = MultiClassBayesNode("concave points_mean", ["diagnosis"], cpts["cpt_concave_points"])
        perimeter_node = MultiClassBayesNode("perimeter_mean", ["diagnosis"], cpts["cpt_perimeter"])
        radius_node = MultiClassBayesNode("radius_mean", ["diagnosis"], cpts["cpt_radius"])
        concavity_node = MultiClassBayesNode("concavity_mean", ["diagnosis", "concave points_mean"], cpts["cpt_concavity"])
        texture_node = MultiClassBayesNode("texture_mean", ["diagnosis"], cpts["cpt_texture"])
        area_node = MultiClassBayesNode("area_mean", ["perimeter_mean"], cpts["cpt_area"])
        compactness_node = MultiClassBayesNode("compactness_mean", ["concavity_mean"], cpts["cpt_compactness"])
        smoothness_node = MultiClassBayesNode("smoothness_mean", ["concavity_mean"], cpts["cpt_smoothness"])
        symmetry_node = MultiClassBayesNode("symmetry_mean", ["compactness_mean"], cpts["cpt_symmetry"])
        fractal_node = MultiClassBayesNode("fractal_dimension_mean", ["symmetry_mean"], cpts["cpt_fractal"])

        cancer_bn = BayesNet([
            diagnosis_node,
            concave_points_node,
            perimeter_node,
            radius_node,
            concavity_node,
            texture_node,
            area_node,
            compactness_node,
            smoothness_node,
            symmetry_node,
            fractal_node
        ])
    except KeyError as e:
        st.error(f"CPT key error: {e}")
        st.stop()

    # Variables used by the network (excluding diagnosis)
    evidence_vars = [
        "concave points_mean",
        "perimeter_mean",
        "radius_mean",
        "concavity_mean",
        "texture_mean",
        "area_mean",
        "compactness_mean",
        "smoothness_mean",
        "symmetry_mean",
        "fractal_dimension_mean"
    ]

    # Ask user for continuous values with short descriptions
    user_inputs = {}
    user_inputs["radius_mean"] = st.number_input("radius_mean (mean radius):", min_value=0.0, step=0.1)
    user_inputs["texture_mean"] = st.number_input("texture_mean (std dev of gray-scale):", min_value=0.0, step=0.1)
    user_inputs["perimeter_mean"] = st.number_input("perimeter_mean (mean size of core tumor):", min_value=0.0, step=0.1)
    user_inputs["area_mean"] = st.number_input("area_mean (mean area of tumor):", min_value=0.0, step=0.1)
    user_inputs["smoothness_mean"] = st.number_input("smoothness_mean (local variation in radius lengths):", min_value=0.0, step=0.001)
    user_inputs["compactness_mean"] = st.number_input("compactness_mean (mean of perimeter^2/area-1):", min_value=0.0, step=0.001)
    user_inputs["concavity_mean"] = st.number_input("concavity_mean (severity of concave portions):", min_value=0.0, step=0.001)
    user_inputs["concave points_mean"] = st.number_input("concave points_mean (number of concave portions):", min_value=0.0, step=0.001)
    user_inputs["symmetry_mean"] = st.number_input("symmetry_mean (symmetry measure):", min_value=0.0, step=0.001)
    user_inputs["fractal_dimension_mean"] = st.number_input("fractal_dimension_mean (coastline approximation):", min_value=0.0, step=0.001)

    def discretize_value(value, bin_edges):
        import numpy as np
        idx = np.digitize([value], bin_edges[1:-1])  
        return int(idx[0])  # bin index

    # Discretize all user inputs using the saved bin edges
    discretized_evidence = {}
    for var in evidence_vars:
        edges = cancer_bins[var]
        val = user_inputs[var]
        bin_idx = discretize_value(val, edges)
        discretized_evidence[var] = bin_idx

    if st.button("Get Cancer Diagnosis"):
        try:
            result = enumeration_ask("diagnosis", discretized_evidence, cancer_bn)
            prediction = max(result.prob, key=result.prob.get)
            # If your training used 1 for M and 0 for B:
            label = "Malignant (M)" if prediction == 1 else "Benign (B)"
            st.success(f"Most likely diagnosis: {label}")
            # Show probabilities
            st.write("Probabilities:")
            st.write(f"Malignant (M, 1): {result.prob.get(1,0):.4f}")
            st.write(f"Benign (B, 0): {result.prob.get(0,0):.4f}")
        except Exception as e:
            st.error(f"An error occurred during inference: {e}")