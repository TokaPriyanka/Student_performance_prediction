import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Student Performance Predictor")
st.caption("ML + OpenAI (One-line explanation)")

# ---------------- LOAD & TRAIN MODEL ----------------
@st.cache_data
def train_model():
    df = pd.read_csv("StudentsPerformance.csv")

    X = df.drop("performance", axis=1)
    y = df["performance"]

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, _, y_train, _ = train_test_split(
        X, y_enc, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, encoder

model, encoder = train_model()

# ---------------- USER INPUT ----------------
st.subheader("📌 Enter Student Details")

cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, 7.5, key="cgpa")
attendance = st.slider("Attendance (%)", 0, 100, 75, key="att")
internal_marks = st.slider("Internal Marks (out of 100)", 0, 100, 70, key="int")
study_hours = st.slider("Study Hours per day", 0, 12, 3, key="study")
projects = st.slider("Number of Projects", 0, 10, 2, key="proj")

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Performance", key="predict"):
    user_df = pd.DataFrame([{
        "cgpa": cgpa,
        "attendance": attendance,
        "internal_marks": internal_marks,
        "study_hours": study_hours,
        "projects": projects
    }])

    pred = model.predict(user_df)
    label = encoder.inverse_transform(pred)[0]

    st.success(f"🎯 Predicted Performance: **{label.upper()}**")

    # ---------------- AI EXPLANATION (ONE LINE) ----------------
    st.subheader("🤖 AI Explanation")

    api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        st.warning("OpenAI API key not found in Secrets.")
    else:
        try:
            client = OpenAI(api_key=api_key)

            prompt = (
                f"Student performance is {label}. "
                f"Give one short sentence explanation."
            )

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=25
            )

            st.info(response.choices[0].message.content)

        except Exception:
            st.info("AI temporarily unavailable. Please try again.")
