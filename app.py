import streamlit as st
import pandas as pd
import openai
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
st.caption("ML + AI Explanation (Always Available)")

# ---------------- TRAIN MODEL ----------------
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

cgpa = st.number_input("CGPA (out of 10)", 0.0, 10.0, 7.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
internal_marks = st.slider("Internal Marks (out of 100)", 0, 100, 70)
study_hours = st.slider("Study Hours per day", 0, 12, 3)
projects = st.slider("Number of Projects", 0, 10, 2)

# ---------------- FALLBACK AI (RULE BASED) ----------------
def fallback_explanation(label):
    if label == "good":
        return (
            "The student is performing well academically. "
            "Strong consistency and discipline are visible. "
            "Maintaining current study habits will help sustain success."
        )
    elif label == "average":
        return (
            "The student shows average academic performance. "
            "With better time management and regular practice, results can improve. "
            "Focusing on weak subjects will be beneficial."
        )
    else:
        return (
            "The student shows poor academic performance. "
            "Improved attendance and structured study time are required. "
            "Seeking guidance and consistent effort can help significantly."
        )

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Performance"):
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

    st.subheader("🤖 AI Explanation")

    explanation = ""

    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]

        prompt = f"Student performance is {label}. Explain in 2 short sentences."

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            timeout=10
        )

        explanation = response["choices"][0]["message"]["content"]

    except Exception:
        # 🚨 NEVER FAIL – ALWAYS FALLBACK
        explanation = fallback_explanation(label)

    st.info(explanation)
