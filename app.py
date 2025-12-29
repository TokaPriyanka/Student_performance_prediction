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
st.caption("Machine Learning + Gemini AI (Deployed on Streamlit Cloud)")

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

    # ---------------- GEMINI AI EXPLANATION ----------------
    st.subheader("🤖 AI Explanation")

    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")

    if GEMINI_KEY is None:
        st.warning("Gemini API key not found. Please add it in Streamlit Secrets.")
    else:
        try:
            genai.configure(api_key=GEMINI_KEY)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
Explain the student's academic performance in exactly 3 sentences.

Sentence 1: Overall performance level.
Sentence 2: Key contributing factors.
Sentence 3: One improvement suggestion.

Performance: {label}
CGPA: {cgpa}
Attendance: {attendance}%
Internal Marks: {internal_marks}
Study Hours: {study_hours}
Projects: {projects}
"""

            response = gemini_model.generate_content(prompt)
            st.info(response.text)

        except Exception:
            st.info("AI service temporarily unavailable. Please try again.")
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
st.caption("Machine Learning + Gemini AI (Deployed on Streamlit Cloud)")

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

    # ---------------- GEMINI AI EXPLANATION ----------------
    st.subheader("🤖 AI Explanation")

    GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")

    if GEMINI_KEY is None:
        st.warning("Gemini API key not found. Please add it in Streamlit Secrets.")
    else:
        try:
            genai.configure(api_key=GEMINI_KEY)
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""
Explain the student's academic performance in exactly 3 sentences.

Sentence 1: Overall performance level.
Sentence 2: Key contributing factors.
Sentence 3: One improvement suggestion.

Performance: {label}
CGPA: {cgpa}
Attendance: {attendance}%
Internal Marks: {internal_marks}
Study Hours: {study_hours}
Projects: {projects}
"""

            response = gemini_model.generate_content(prompt)
            st.info(response.text)

        except Exception:
            st.info("AI service temporarily unavailable. Please try again.")
