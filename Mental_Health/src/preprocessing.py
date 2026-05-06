"""
preprocessing.py
----------------
Handles all data loading, exploratory data analysis, encoding,
train/test splitting, and imputation for the student mental health project.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer   
 
 
# ── Data Loading ──────────────────────────────────────────────────────────────
 
def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from a CSV file and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
 
 
# ── Exploratory Data Analysis ─────────────────────────────────────────────────
 
def summarise_data(df: pd.DataFrame) -> None:
    """Print structure, data types, descriptive statistics, and missing values."""
    print("=== Data Types and Shape ===")
    df.info()
    print("\n=== Descriptive Statistics ===")
    print(df.describe())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
 
 
def plot_target_distribution(df: pd.DataFrame, target_col: str = "Depression_Score") -> None:
    """Plot the distribution of the target variable."""
    plt.figure(figsize=(6, 4))
    df[target_col].hist(bins=30, color="steelblue", edgecolor="white")
    plt.title("Distribution of Depression Scores")
    plt.xlabel("Depression Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
 
 
def show_sample_sizes(df: pd.DataFrame) -> None:
    """Print sample sizes for Course, Gender, and Age to justify variability in group stats."""
    print("Sample sizes by Course:")
    print(df["Course"].value_counts())
 
    print("\nSample sizes by Gender:")
    print(df["Gender"].value_counts())
 
    print("\nSample sizes by Age:")
    print(df["Age"].value_counts().sort_index())
 
 
def analyse_group(df: pd.DataFrame, group_col: str, target_col: str = "Depression_Score") -> pd.DataFrame:
    """
    Calculate mean, std, and count of the target variable grouped by a column.
    Returns the stats DataFrame and displays a boxplot.
    """
    stats = df.groupby(group_col)[target_col].agg(["mean", "std", "count"]).sort_values("mean")
    print(f"\nDepression Stats by {group_col}:")
    print(stats)
 
    fig_width = max(6, len(stats) * 1.5)
    plt.figure(figsize=(fig_width, 5))
    sns.boxplot(data=df, x=group_col, y=target_col, order=stats.index.tolist())
    plt.title(f"Depression Score Distribution by {group_col}")
    plt.xlabel(group_col)
    plt.ylabel(target_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
 
    return stats
 
 
def analyse_interaction(df: pd.DataFrame, col1: str, col2: str,
                         target_col: str = "Depression_Score") -> None:
    """Print mean target scores for the interaction between two categorical columns."""
    interaction = df.groupby([col1, col2])[target_col].mean()
    print(f"\nDepression by {col1} and {col2}:")
    print(interaction)
 
 
def show_overall_stats(df: pd.DataFrame, target_col: str = "Depression_Score") -> None:
    """Print overall descriptive statistics for the target variable."""
    print("Overall Depression Stats:")
    print(df[target_col].describe())
 
 
def detect_outliers(df: pd.DataFrame) -> dict:
    """
    Detect outliers in all numeric columns using the IQR method.
    Returns a dictionary of column -> outlier count.
    Outliers are flagged but retained as they represent valid real-world values
    (e.g. mature students aged 30-35, students with extreme CGPA scores).
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    outlier_counts = {}
 
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_counts[col] = len(df[(df[col] < lower) | (df[col] > upper)])
 
    print("Outlier counts per column (IQR method):")
    for col, count in outlier_counts.items():
        print(f"  {col}: {count}")
 
    print("\nDecision: All outliers retained - values are real and domain-valid.")
    print("Age outliers (30-35) represent mature students.")
    print("CGPA outliers represent students at the extremes of academic performance.")
 
    return outlier_counts
 
 
# ── Encoding ──────────────────────────────────────────────────────────────────
 
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical columns using drop_first=True.
    Encoding is done before correlation analysis so categorical variables
    such as Course and Gender are included in the correlation matrix.
    Returns the encoded DataFrame.
    """

    sleep_map = {
        "Poor": 0,
        "Moderate": 1,
        "Good": 2
    }
    df["Sleep_Quality_Ordinal"] = df["Sleep_Quality"].map(sleep_map)

    df["Mental_Strain"] = (
        df["Stress_Level"] +
        df["Anxiety_Score"] +
        df["Financial_Stress"]
    )

    df["Stress_Sleep_Interaction"] = df["Stress_Level"] * df["Sleep_Quality_Ordinal"]
    df["Anxiety_Financial_Interaction"] = df["Anxiety_Score"] * df["Financial_Stress"]
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    print(f"Encoded dataframe shape: {df_encoded.shape}")
    return df_encoded
 
 
def show_correlation(df_encoded: pd.DataFrame, target_col: str = "Depression_Score") -> None:
    """Print correlation of all features with the target variable."""
    corr = df_encoded.corr(numeric_only=True)[target_col].sort_values(ascending=False)
    print(f"\nCorrelation with {target_col} (encoded):")
    print(corr)
 
 
# ── Splitting and Imputation ──────────────────────────────────────────────────
 
def split_data(df_encoded: pd.DataFrame, target_col: str = "Depression_Score",
               test_size: float = 0.2, random_state: int = 42):
    """
    Split the encoded DataFrame into train and test sets.
    A single global split is used by all models to ensure
    R² scores are directly comparable across models.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train size: {X_train.shape[0]} rows | Test size: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test
 
 
def impute_missing(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Impute missing numeric values using the mean strategy.
    The imputer is fitted on X_train only to prevent data leakage.
    X_test is transformed using training statistics only.
    Returns the imputed X_train and X_test.
    """
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
 
    imputer = SimpleImputer(strategy="mean")
    X_train[numeric_features] = imputer.fit_transform(X_train[numeric_features].copy())
    X_test[numeric_features] = imputer.transform(X_test[numeric_features].copy())
 
    remaining_train = X_train.isnull().sum().sum()
    remaining_test = X_test.isnull().sum().sum()
    print(f"Imputation complete. Remaining nulls - Train: {remaining_train} | Test: {remaining_test}")
 
    return X_train, X_test