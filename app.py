import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Student Performance Predictor")
st.caption("Machine Learning + Google Gemini (Free Tier)")

# ---------------- LOAD & TRAIN MODEL ----------------
@st.cache_data
def train_model():
    df = pd.read_csv("StudentsPerformance.csv")

    X = df.drop("performance", axis=1)
    y = df["performance"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, _, y_train, _ = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, encoder

model, encoder = train_model()

# ---------------- USER INPUT ----------------
st.subheader("📌 Enter Student Details")

cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, 7.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
internal_marks = st.slider("Internal Marks (out of 100)", 0, 100, 70)
study_hours = st.slider("Study Hours per day", 0, 12, 3)
projects = st.slider("Number of Projects", 0, 10, 2)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Performance"):
    user_data = pd.DataFrame([{
        "cgpa": cgpa,
        "attendance": attendance,
        "internal_marks": internal_marks,
        "study_hours": study_hours,
        "projects": projects
    }])

    prediction = model.predict(user_data)
    label = encoder.inverse_transform(prediction)[0]

    st.success(f"🎯 Predicted Performance: **{label.upper()}**")

    # ---------------- AI EXPLANATION ----------------
    st.subheader("🤖 AI Explanation")

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        gemini = genai.GenerativeModel("gemini-pro")

        prompt = f"""
A student's academic performance is classified as {label}.
CGPA: {cgpa}, Attendance: {attendance}%, Internal Marks: {internal_marks},
Study Hours: {study_hours}, Projects: {projects}.

Explain the performance in 2–3 short academic sentences.
"""

        with st.spinner("Generating AI explanation..."):
            response = gemini.generate_content(prompt)
            st.write(response.text)

    except Exception:
        st.info("AI service temporarily busy. Please try again in a moment.")
