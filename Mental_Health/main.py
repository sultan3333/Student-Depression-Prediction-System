"""
main.py
-------
Entry point for the Student Mental Health Depression Score Prediction project.
Handles data loading, EDA, preprocessing, model training, evaluation, and saving
the best model for later use in the Gradio interface.
 
Run from project root:
    python main.py
"""

import os
import joblib

from src.preprocessing import (
    load_data,
    summarise_data,
    plot_target_distribution,
    show_sample_sizes,
    analyse_group,
    analyse_interaction,
    show_overall_stats,
    detect_outliers,
    encode_features,
    show_correlation,
    split_data,
    impute_missing,
)

from src.models import (
    train_all_models,
    train_single_factor_models,
)

from src.evaluation import (
    evaluate_all_models,
    evaluate_single_factor_models,
    plot_model_comparison,
    plot_single_vs_multi,
    plot_feature_importance,
)


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "students_mental_health_survey.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")

TARGET_COL     = "Depression_Score"
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

SINGLE_FACTORS = ["Stress_Level", "Anxiety_Score", "Financial_Stress", "CGPA", "Age"]


# ── Pipeline ──────────────────────────────────────────────────────────────────

def main():

    # 1. Load data
    print("\n[1/7] Loading data...")
    df = load_data(DATA_PATH)

    # 2. Exploratory Data Analysis
    print("\n[2/7] Exploratory Data Analysis...")
    summarise_data(df)
    plot_target_distribution(df, TARGET_COL)
    show_sample_sizes(df)
    analyse_group(df, "Course", TARGET_COL)
    analyse_group(df, "Gender", TARGET_COL)
    analyse_group(df, "Age", TARGET_COL)
    analyse_interaction(df, "Gender", "Course", TARGET_COL)
    show_overall_stats(df, TARGET_COL)

    # 3. Outlier detection
    print("\n[3/7] Outlier detection...")
    detect_outliers(df)

    # 4. Encoding and correlation
    print("\n[4/7] Encoding features and computing correlation...")
    df_encoded = encode_features(df)
    show_correlation(df_encoded, TARGET_COL)

    # 5. Split and impute
    print("\n[5/7] Splitting and imputing...")
    X_train, X_test, y_train, y_test = split_data(
        df_encoded, TARGET_COL, TEST_SIZE, RANDOM_STATE
    )
    X_train, X_test = impute_missing(X_train, X_test)

    # 6. Train and evaluate models
    print("\n[6/7] Training and evaluating models...")
    models = train_all_models(X_train, y_train)

    df_results = evaluate_all_models(models, X_test, y_test)

    print("\nModel Comparison Table:")
    print(df_results.to_string())

    plot_model_comparison(df_results)

    # Feature importance
    plot_feature_importance(models["random_forest"], X_train)

    # Single-factor models
    single_models = train_single_factor_models(
        X_train, X_test, y_train, y_test, SINGLE_FACTORS
    )

    df_single = evaluate_single_factor_models(single_models, X_test, y_test)
    plot_single_vs_multi(df_single, df_results)

    best_model_name = df_results.sort_values("R2", ascending=False).index[0]
    best_model = models[best_model_name]
    # 7. SAVE BEST MODEL (IMPORTANT FOR GRADIO)
    print("\n[7/7] Saving best model...")

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    model_bundle = {    
    "model": best_model,
    "columns": X_train.columns.tolist()
}

    joblib.dump(model_bundle, MODEL_PATH)

    print(f"Model saved at: {MODEL_PATH}")
    print(f"Best model selected: {best_model_name}")


if __name__ == "__main__":
    main()