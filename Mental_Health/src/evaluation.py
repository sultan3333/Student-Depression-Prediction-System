"""
evaluation.py
-------------
Handles all model evaluation, results reporting, visualisation,
and the chatbot prediction interface for the student mental health project.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import LinearRegression
 
 
# ── Metric Computation ────────────────────────────────────────────────────────
 
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str) -> dict:
    """
    Evaluate a single trained model on the test set.
    Reports R² and RMSE. Returns a dict with results.
 
    Metrics:
      R²   = 1 - (SS_res / SS_tot)  — proportion of variance explained
      RMSE = sqrt(mean((y - y_hat)^2)) — average prediction error in original units
    """
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
 
    print(f"{model_name}:")
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
 
    return {"model": model_name, "R2": round(r2, 4), "RMSE": round(rmse, 4)}
 
 
def evaluate_all_models(models: dict, X_test: pd.DataFrame,
                        y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate all models in a dictionary and return a comparison DataFrame.
    Models dict expected: {'linear': model, 'ridge': model, ...}
    """
    label_map = {
        "linear":        "linear",
        "ridge":         "ridge",
        "lasso":         "lasso",
        "random_forest": "random forest",
    }
 
    results = []
    for key, model in models.items():
        label = label_map.get(key, key)
        result = evaluate_model(model, X_test, y_test, label)
        results.append(result)
 
    df_results = pd.DataFrame(results).set_index("model")
    return df_results
 
 
def evaluate_single_factor_models(single_models: dict, X_test: pd.DataFrame,
                                   y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate all single-factor models and return a results DataFrame.
    Used to compare individual feature predictive power against multi-factor models.
    """
    results = []
    print("Single-Factor R² Scores (Linear Regression):")
    for factor, model in single_models.items():
        y_pred = model.predict(X_test[[factor]])
        r2   = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        print(f"  {factor}: R² = {r2:.4f} | RMSE = {rmse:.4f}")
        results.append({"factor": factor, "R2": round(r2, 4), "RMSE": round(rmse, 4)})
 
    return pd.DataFrame(results).set_index("factor")
    
 
# ── Visualisation ─────────────────────────────────────────────────────────────
 
def plot_model_comparison(df_results: pd.DataFrame) -> None:
    """
    Bar chart comparing R² scores across all multi-factor models.
    Lower RMSE and higher R² indicate better performance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    df_results["R2"].plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
    axes[0].set_title("Model Comparison - R²")
    axes[0].set_ylabel("R² Score")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(0, max(df_results["R2"].max() * 1.2, 0.1))
 
    df_results["RMSE"].plot(kind="bar", ax=axes[1], color="salmon", edgecolor="white")
    axes[1].set_title("Model Comparison - RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=30)
 
    plt.suptitle("Multi-Factor Model Performance Comparison", fontsize=13)
    plt.tight_layout()
    plt.show()
 
 
def plot_single_vs_multi(df_single: pd.DataFrame, df_multi: pd.DataFrame) -> None:
    """
    Bar chart comparing single-factor R² scores against the best multi-factor model.
    Illustrates that individual features have very weak predictive power,
    and combining features only marginally improves prediction.
    """
    best_multi_r2 = df_multi["R2"].max()
    best_multi_name = df_multi["R2"].idxmax()
 
    ax = df_single["R2"].plot(kind="bar", figsize=(10, 5),
                               color="steelblue", edgecolor="white",
                               label="Single-Factor Models")
    ax.axhline(y=best_multi_r2, color="red", linestyle="--", linewidth=1.5,
               label=f"Best Multi-Factor ({best_multi_name}): R² = {best_multi_r2:.4f}")
    ax.set_title("Single-Factor vs Best Multi-Factor Model (R²)")
    ax.set_ylabel("R² Score")
    ax.set_xlabel("Feature")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    plt.tight_layout()
    plt.show()
 
 
def plot_feature_importance(rf_model, X_train: pd.DataFrame, top_n: int = 10) -> None:
    """
    Plot the top N feature importances from the Random Forest model.
    Helps identify which variables contribute most to the model's predictions.
    """
    importance = pd.Series(rf_model.feature_importances_, index=X_train.columns)
    importance.sort_values(ascending=False).head(top_n).plot(
        kind="bar", figsize=(10, 5), color="steelblue", edgecolor="white"
    )
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.ylabel("Importance Score")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.show()
 
 
# ── Chatbot Prediction Interface ──────────────────────────────────────────────
 
def score_band(score: float) -> tuple:
    """
    Return a risk band label and associated colour for a given depression score.
    Bands: Low (0.0-1.5), Moderate (1.5-3.0), High (3.0-5.0)
    """
    if score <= 1.5:
        return "Low", "#2ecc71"
    elif score <= 3.0:
        return "Moderate", "#f39c12"
    else:
        return "High", "#e74c3c"
 
 
def build_input_row(course: str, gender: str, age: int, cgpa: float,
                    stress: int, anxiety: int, financial: int, sleep_quality: str,
                    X_train_columns: pd.Index) -> pd.DataFrame:
    """
    Build a single-row input DataFrame that matches the encoded training feature space.
    Recreates the one-hot encoding produced by pd.get_dummies(drop_first=True)
    so the model receives the correct feature vector format.
    """
    sleep_map = {"Poor": 0, "Moderate": 1, "Good": 2}
    sleep_ordinal = sleep_map.get(sleep_quality, 1)

    row = {
        "Age":                          age,
        "CGPA":                         cgpa,
        "Stress_Level":                 stress,
        "Anxiety_Score":                anxiety,
        "Financial_Stress":             financial,
        "Sleep_Quality_Ordinal":        sleep_ordinal,
        "Mental_Strain":                stress + anxiety + financial,
        "Stress_Sleep_Interaction":     stress * sleep_ordinal,
        "Anxiety_Financial_Interaction": anxiety * financial,
    }
 
    for c in ["Computer Science", "Engineering", "Law", "Medical", "Others"]:
        row[f"Course_{c}"] = int(course == c)
 
    row["Gender_Male"] = int(gender == "Male")
 
    full_row = pd.DataFrame([row])
    full_row = full_row.reindex(columns=X_train_columns, fill_value=0)
    return full_row
 
 
def predict_depression_score(model, course: str, gender: str, age: int,
                              cgpa: float, stress: int, anxiety: int,
                              financial: int, X_train_columns: pd.Index) -> None:
    """
    Run a prediction and print a formatted result summary.
    Clamps the predicted score to [0, 5] and displays the risk band,
    a context message, and a clinical disclaimer.
    """
    row = build_input_row(course, gender, age, cgpa, stress, anxiety,
                          financial, X_train_columns)
 
    pred = float(model.predict(row)[0])
    pred = round(max(0.0, min(5.0, pred)), 2)
    band, _ = score_band(pred)
    dataset_mean = 2.25
 
    if course == "Computer Science":
        context = ("Computer Science students show the highest average "
                   "depression score in this dataset (3.30 vs overall mean 2.25).")
    elif pred < dataset_mean:
        context = f"Your predicted score is below the dataset average of {dataset_mean}."
    else:
        context = f"Your predicted score is above the dataset average of {dataset_mean}."
 
    print("=" * 52)
    print("       STUDENT MENTAL HEALTH - PREDICTION")
    print("=" * 52)
    print(f"  Predicted Depression Score : {pred:.2f} / 5.00")
    print(f"  Risk Band                  : {band}")
    print(f"  Dataset Average            : {dataset_mean}")
    print("-" * 52)
    print(f"  {context}")
    print("=" * 52)
    print()
    print("  Bands:  Low 0.0-1.5  |  Moderate 1.5-3.0  |  High 3.0-5.0")
    print()
    print("  DISCLAIMER: This prediction is generated by a")
    print("  machine-learning model trained on survey data.")
    print("  It is NOT a clinical diagnosis. If you are")
    print("  struggling, please speak to a professional.")
    print("=" * 52)
 
 
def run_chatbot(model, X_train_columns: pd.Index) -> None:
    """
    Launch an interactive text-based chatbot in the terminal or notebook.
    Prompts the user for inputs and returns a depression score prediction.
    Uses plain input() calls - no external widget libraries required.
    """
    print("\n" + "=" * 52)
    print("   STUDENT DEPRESSION SCORE PREDICTOR")
    print("=" * 52)
 
    courses = ["Business", "Computer Science", "Engineering", "Law", "Medical", "Others"]
    print("\nCourses available:")
    for i, c in enumerate(courses, 1):
        print(f"  {i}. {c}")
    course_idx = int(input("Select course number: ")) - 1
    course = courses[course_idx]
 
    gender = input("Gender (Male/Female): ").strip().capitalize()
    age = int(input("Age (18-35): ").strip())
    cgpa = float(input("CGPA (0.0-4.0): ").strip())
    stress = int(input("Stress Level (0-5): ").strip())
    anxiety = int(input("Anxiety Score (0-5): ").strip())
    financial = int(input("Financial Stress (0-5): ").strip())
 
    predict_depression_score(
        model=model,
        course=course,
        gender=gender,
        age=age,
        cgpa=cgpa,
        stress=stress,
        anxiety=anxiety,
        financial=financial,
        X_train_columns=X_train_columns
    )