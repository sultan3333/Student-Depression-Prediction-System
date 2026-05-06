"""
app.py
------
Gradio-based web interface for the Student Mental Health Depression
Score Predictor. Loads the pre-trained model bundle saved by main.py
and provides an interactive browser-based prediction form.

Run from the Mental_Health root folder:
    python src/app.py
"""

import gradio as gr
import pandas as pd
import joblib
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

# ── Load model bundle saved by main.py ───────────────────────────────────────
bundle        = joblib.load(MODEL_PATH)
model         = bundle["model"]
TRAIN_COLUMNS = bundle["columns"]


# ── Input builder ─────────────────────────────────────────────────────────────
def build_input(course, gender, sleep_quality, age, cgpa,
                stress, anxiety, financial):
    """
    Builds a single-row DataFrame that exactly matches the feature space
    used during training, including all engineered features created in
    preprocessing.py.
    """
    # Ordinal encoding for Sleep Quality — must match preprocessing.py
    sleep_map = {"Poor": 0, "Moderate": 1, "Good": 2}
    sleep_ordinal = sleep_map.get(sleep_quality, 1)

    row = {
        "Age":                           age,
        "CGPA":                          cgpa,
        "Stress_Level":                  stress,
        "Anxiety_Score":                 anxiety,
        "Financial_Stress":              financial,
        # Engineered features — must match preprocessing.py exactly
        "Sleep_Quality_Ordinal":         sleep_ordinal,
        "Mental_Strain":                 stress + anxiety + financial,
        "Stress_Sleep_Interaction":      stress * sleep_ordinal,
        "Anxiety_Financial_Interaction": anxiety * financial,
    }

    # One-hot encoding for Course (reference category = Business)
    for c in ["Computer Science", "Engineering", "Law", "Medical", "Others"]:
        row[f"Course_{c}"] = 1 if course == c else 0

    # One-hot encoding for Gender (reference category = Female)
    row["Gender_Male"] = 1 if gender == "Male" else 0

    df = pd.DataFrame([row])

    # Align exactly with training feature order and fill any missing cols with 0
    df = df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

    return df


# ── Prediction logic ──────────────────────────────────────────────────────────
def predict(course, gender, sleep_quality, age, cgpa,
            stress, anxiety, financial):
    """
    Runs the trained model on user inputs and returns a formatted
    prediction string with score, risk band, and disclaimer.
    """
    X    = build_input(course, gender, sleep_quality, age, cgpa,
                       stress, anxiety, financial)
    pred = model.predict(X)[0]
    pred = round(max(0.0, min(5.0, pred)), 2)

    if pred <= 1.5:
        risk = "Low"
    elif pred <= 3.0:
        risk = "Moderate"
    else:
        risk = "High"

    dataset_mean = 2.25
    if course == "Computer Science":
        context = (
            "Computer Science students have the highest average depression "
            "score in this dataset (3.30 vs overall mean 2.25)."
        )
    elif pred < dataset_mean:
        context = f"Your predicted score is below the dataset average of {dataset_mean}."
    else:
        context = f"Your predicted score is above the dataset average of {dataset_mean}."

    return (
        f"Predicted Score : {pred} / 5.00\n"
        f"Risk Level      : {risk}\n\n"
        f"{context}\n\n"
        f"Bands: Low 0.0-1.5  |  Moderate 1.5-3.0  |  High 3.0-5.0\n\n"
        f"DISCLAIMER: This is a machine-learning prediction based on survey\n"
        f"data. It is NOT a clinical diagnosis. If you are struggling,\n"
        f"please speak to a medical professional."
    )


# ── About section ─────────────────────────────────────────────────────────────
about_section = """
## Student Mental Health Prediction System

This application predicts a student's depression score based on academic,
psychological, and demographic factors.

The model uses machine learning regression techniques trained on survey data
to estimate mental health severity on a scale from 0 to 5.

### Limitations:
- This model is not a clinical diagnostic tool
- Predictions are based on historical data patterns and may not be fully accurate
- Results should not replace professional medical advice
"""


# ── Gradio Interface ──────────────────────────────────────────────────────────
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(
            choices=["Business", "Computer Science", "Engineering",
                     "Law", "Medical", "Others"],
            label="Course",
            value="Business"
        ),
        gr.Radio(
            choices=["Female", "Male"],
            label="Gender",
            value="Female"
        ),
        gr.Dropdown(
            choices=["Poor", "Moderate", "Good"],
            label="Sleep Quality",
            value="Moderate"
        ),
        gr.Number(label="Age", value=20, minimum=18, maximum=35),
        gr.Number(label="CGPA", value=3.0, minimum=0.0, maximum=4.0),
        gr.Slider(minimum=0, maximum=5, step=1, value=2, label="Stress Level"),
        gr.Slider(minimum=0, maximum=5, step=1, value=2, label="Anxiety Score"),
        gr.Slider(minimum=0, maximum=5, step=1, value=2, label="Financial Stress"),
    ],
    outputs=gr.Textbox(label="Prediction Result", lines=10),
    title="Student Mental Health Prediction System",
    description=about_section,
)

if __name__ == "__main__":
    interface.launch(theme=gr.themes.Soft())