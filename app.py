"""
Diabetes Detection — Streamlit App
-------------------------------------
Interactive web app that loads the trained model and lets a user enter
patient details to get a live diabetes risk prediction, alongside the
dataset's EDA and model performance metrics.

Run with:
    streamlit run app.py
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# --------------------------------------------------------------------------
# Load model artifacts (cached so they load once per session)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/diabetes_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    with open("model/metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


@st.cache_data
def load_dataset():
    return pd.read_csv("data/diabetes.csv")


model, scaler, metadata = load_artifacts()
df = load_dataset()

encoders = metadata["encoders"]
feature_cols = metadata["feature_cols"]
uses_scaled_input = metadata["uses_scaled_input"]

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.title("🩺 Diabetes Detection")
page = st.sidebar.radio("Navigate", ["Predict", "Dataset Overview", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Best model: **{metadata['best_model']}**  \n"
    f"F1-Score: **{metadata['results'][metadata['best_model']]['f1']}**"
)

# --------------------------------------------------------------------------
# PAGE 1 — Prediction
# --------------------------------------------------------------------------
if page == "Predict":
    st.title("Diabetes Risk Predictor")
    st.write(
        "Enter patient details below to estimate diabetes risk using a "
        f"**{metadata['best_model']}** model trained on {len(df):,} patient records."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 80, 45)
        bmi = st.number_input("BMI", min_value=16.0, max_value=45.0, value=26.0, step=0.1)
        blood_pressure = st.slider("Blood Pressure", 85, 190, 120)

    with col2:
        glucose = st.slider("Glucose Level", 65, 250, 120)
        insulin = st.slider("Insulin Level", 20, 261, 100)
        gender = st.selectbox("Gender", encoders["Gender"])

    with col3:
        physical_activity = st.selectbox("Physical Activity", encoders["Physical_Activity"])
        smoking = st.selectbox("Smoking", encoders["Smoking"])
        family_history = st.selectbox("Family History of Diabetes", encoders["Family_History"])
        high_bp = st.selectbox("High Blood Pressure", encoders["High_Blood_Pressure"])

    if st.button("Predict Diabetes Risk", type="primary", use_container_width=True):
        input_dict = {
            "Age": age,
            "BMI": bmi,
            "Blood_Pressure": blood_pressure,
            "Glucose_Level": glucose,
            "Insulin_Level": insulin,
            "Gender": encoders["Gender"].index(gender),
            "Physical_Activity": encoders["Physical_Activity"].index(physical_activity),
            "Smoking": encoders["Smoking"].index(smoking),
            "Family_History": encoders["Family_History"].index(family_history),
            "High_Blood_Pressure": encoders["High_Blood_Pressure"].index(high_bp),
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]

        if uses_scaled_input:
            input_array = scaler.transform(input_df)
        else:
            input_array = input_df

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        st.markdown("---")
        result_col, gauge_col = st.columns([1, 1])

        with result_col:
            if prediction == 1:
                st.error(f"### ⚠️ Higher Diabetes Risk\nEstimated probability: **{probability:.1%}**")
            else:
                st.success(f"### ✅ Lower Diabetes Risk\nEstimated probability: **{probability:.1%}**")
            st.caption(
                "This is a demo model for portfolio purposes, not a medical diagnosis. "
                "Please consult a healthcare professional for medical advice."
            )

        with gauge_col:
            fig = px.pie(
                values=[probability, 1 - probability],
                names=["Risk", "No Risk"],
                hole=0.6,
                color_discrete_sequence=["#EF553B", "#636EFA"],
            )
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 2 — Dataset Overview / EDA
# --------------------------------------------------------------------------
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write(f"**{len(df):,} patient records** · **{df.shape[1]} columns**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Patients", f"{len(df):,}")
    m2.metric("Diabetic", f"{(df['Diabetes'] == 'Yes').sum():,}")
    m3.metric("Non-Diabetic", f"{(df['Diabetes'] == 'No').sum():,}")
    m4.metric("Diabetes Rate", f"{(df['Diabetes'] == 'Yes').mean():.1%}")

    st.subheader("Class Balance")
    class_counts = df["Diabetes"].value_counts().reset_index()
    class_counts.columns = ["Diabetes", "Count"]
    st.plotly_chart(
        px.bar(class_counts, x="Diabetes", y="Count", color="Diabetes", text="Count"),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Glucose Level Distribution")
        st.plotly_chart(
            px.histogram(df, x="Glucose_Level", color="Diabetes", barmode="overlay", nbins=40),
            use_container_width=True,
        )
    with c2:
        st.subheader("BMI vs Age")
        st.plotly_chart(
            px.scatter(df, x="Age", y="BMI", color="Diabetes", opacity=0.5),
            use_container_width=True,
        )

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50), use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 3 — Model Performance
# --------------------------------------------------------------------------
else:
    st.title("Model Performance")
    st.write("Comparison of the two classifiers trained on this dataset.")

    results_df = pd.DataFrame(metadata["results"]).T.reset_index()
    results_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.plotly_chart(
        px.bar(
            results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            range_y=[0, 1],
        ),
        use_container_width=True,
    )

    st.info(
        f"**{metadata['best_model']}** was selected as the deployed model based on "
        "the highest F1-Score, which balances precision and recall — important for "
        "medical screening where both false positives and false negatives carry a cost."
    )
