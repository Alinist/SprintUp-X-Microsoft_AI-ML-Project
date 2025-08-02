import streamlit as st
import pandas as pd
import joblib
import seaborn as sns

# -------------------------------
# 🔧 Configuration & Model Loading
# -------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction with PCA & ML")

# Column definitions
numerical_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# Load models and preprocessors
pca = joblib.load("../models/pca_transformer.pkl")
model = joblib.load("../models/best_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
encoders = joblib.load("../models/label_encoders.pkl")

# ----------------------------------
# 🧍 User Input via Sidebar
# ----------------------------------
st.sidebar.header("🩺 Patient Information")

# Display-friendly to encoder value mapping
mapping_dicts = {
    "sex": {"Male": "Male", "Female": "Female"},
    "cp": {
        "Typical Angina": "typical angina",
        "Atypical Angina": "atypical angina",
        "Non-Anginal Pain": "non-anginal",
        "Asymptomatic": "asymptomatic",
    },
    "fbs": {"True": "True", "False": "False"},
    "restecg": {
        "Normal": "normal",
        "ST-T Abnormality": "st-t abnormality",
        "Left Ventricular Hypertrophy": "lv hypertrophy"
    },
    "exang": {"True": "True", "False": "False"},
    "slope": {
        "Upsloping": "upsloping",
        "Flat": "flat",
        "Downsloping": "downsloping"
    },
    "thal": {
        "Normal": "normal",
        "Fixed Defect": "fixed defect",
        "Reversable Defect": "reversable defect"
    }
}


# Step 1: Get user input using friendly labels
input_data = {
    'age': st.sidebar.number_input("Age", min_value=1, max_value=120, value=50),
    'sex': st.sidebar.selectbox("Sex", options=list(mapping_dicts["sex"].keys())),
    'cp': st.sidebar.selectbox("Chest Pain Type", options=list(mapping_dicts["cp"].keys())),
    'trestbps': st.sidebar.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120),
    'chol': st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200),
    'fbs': st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=list(mapping_dicts["fbs"].keys())),
    'restecg': st.sidebar.selectbox("Resting ECG", options=list(mapping_dicts["restecg"].keys())),
    'thalch': st.sidebar.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150),
    'exang': st.sidebar.selectbox("Exercise Induced Angina", options=list(mapping_dicts["exang"].keys())),
    'oldpeak': st.sidebar.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1),
    'slope': st.sidebar.selectbox("Slope of ST Segment", options=list(mapping_dicts["slope"].keys())),
    'ca': st.sidebar.number_input("Number of Major Vessels (0–3)", min_value=0, max_value=4, value=0),
    'thal': st.sidebar.selectbox("Thalassemia", options=list(mapping_dicts["thal"].keys())),
}

# Step 2: Map human-friendly input to model format
for col, mapping in mapping_dicts.items():
    if col in input_data:
        input_data[col] = mapping[input_data[col]]


input_df = pd.DataFrame([input_data])

# ----------------------------------
# 🧼 Preprocessing Input Data
# ----------------------------------
# Scale numerical features
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Encode categorical features
for col in categorical_cols:
    val_str = str(input_df.at[0, col])
    try:
        input_df[col] = encoders[col].transform([val_str])
    except ValueError as e:
        st.error(f"❌ Invalid value for '{col}': '{val_str}'\n{e}")

# ----------------------------------
# 🤖 Predict with PCA + Model
# ----------------------------------
input_pca = pca.transform(input_df)
prediction = model.predict(input_pca)[0]
decision_score = model.decision_function(input_pca)[0]

st.subheader("🔍 Prediction Result")
if prediction == 1:
    st.error(f"Prediction: **Heart Disease Detected**")
else:
    st.success(f"Prediction: **No Heart Disease Detected**")
st.info(f"Model Confidence Score: `{decision_score:.2f}`")

# ----------------------------------
# 📊 Data Visualization Section
# ----------------------------------
st.subheader("📊 Heart Disease Trends Explorer")

# Load raw, cleaned dataset for visualization (non-transformed)
data = pd.read_csv("../data/visualized_heart_disease.csv")

tab1, tab2, tab3 = st.tabs(["Age vs Disease", "Cholesterol Distribution", "Interactive Explorer"])

# Age Histogram
with tab1:
    st.markdown("#### Age Distribution by Disease Presence")
    fig1 = sns.histplot(data=data, x='age', hue='target', kde=True, element='step')
    st.pyplot(fig1.figure)

# Cholesterol Boxplot
with tab2:
    st.markdown("#### Cholesterol Levels by Disease Status")
    fig2 = sns.boxplot(data=data, x='target', y='chol')
    st.pyplot(fig2.figure)

# Interactive Filter + Heatmap
with tab3:
    st.markdown("#### Filter by Age")
    age_filter = st.slider("Select Age Range", int(data['age'].min()), int(data['age'].max()), (40, 60))
    filtered = data[(data['age'] >= age_filter[0]) & (data['age'] <= age_filter[1])]
    st.write(f"Showing **{len(filtered)}** patients between ages {age_filter[0]} and {age_filter[1]}")
    st.dataframe(filtered)

    st.markdown("#### Correlation Heatmap (Numeric Only)")
    numeric_filtered = filtered.select_dtypes(include='number')
    fig3 = sns.heatmap(numeric_filtered.corr(), annot=True, cmap='coolwarm')
    st.pyplot(fig3.figure)
