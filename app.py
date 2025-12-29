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
st.caption("Machine Learning + Gemini AI")

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

cgpa = st.number_input(
    "CGPA (out of 10)", 0.0, 10.0, 7.5, key="cgpa"
)

attendance = st.slider(
    "Attendance (%)", 0, 100, 75, key="attendance"
)

internal_marks = st.slider(
    "Internal Marks (out of 100)", 0, 100, 70, key="internal_marks"
)

study_hours = st.slider(
    "Study Hours per day", 0, 12, 3, key="study_hours"
)

projects = st.slider(
    "Number of Projects", 0, 10, 2, key="projects"
)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Performance", key="predict_btn"):
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

    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")

    if not GEMINI_KEY:
        st.warning("Gemini API key not found in Streamlit Secrets.")
    else:
        try:
            genai.configure(api_key=GEMINI_KEY)
            gemini = genai.GenerativeModel("gemini-1.5-flash")

            prompt = (
                f"The student's performance is {label}. "
                f"CGPA is {cgpa}, attendance is {attendance}%. "
                f"Give a short explanation and one improvement suggestion."
            )

            response = gemini.generate_content(prompt)
            st.info(response.text)

        except Exception:
            st.info("AI service temporarily unavailable. Please try again.")
