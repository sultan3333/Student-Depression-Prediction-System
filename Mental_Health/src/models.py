"""
models.py
---------
Defines, trains, and returns all regression models used in the
student mental health depression score prediction project.
"""
 
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
 
 
# ── Model Training ────────────────────────────────────────────────────────────
 
def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a standard Linear Regression model.
    Used as the primary baseline model and for single-factor comparisons.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
 
 
def train_ridge(X_train: pd.DataFrame, y_train: pd.Series,
                alpha: float = 1.0) -> Ridge:
    """
    Train a Ridge Regression model with L2 regularisation.
    Included to assess whether penalising large coefficients improves
    generalisation over plain Linear Regression.
    Alpha controls regularisation strength (default 1.0).
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model
 
 
def train_lasso(X_train: pd.DataFrame, y_train: pd.Series,
                alpha: float = 0.1) -> Lasso:
    """
    Train a Lasso Regression model with L1 regularisation.
    L1 regularisation can drive some coefficients to exactly zero,
    providing implicit feature selection.
    Alpha is set lower than Ridge (0.1) as Lasso is more aggressive.
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model
 
 
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        n_estimators: int = 300, random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor.
    Selected as a non-linear ensemble method to test whether complex
    feature interactions can improve predictive performance beyond linear models.
    Uses 300 trees for stability; random_state ensures reproducibility.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model
 
 
# ── Single-Factor Models ──────────────────────────────────────────────────────
 
def train_single_factor_models(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                factors: list) -> dict:
    """
    Train individual Linear Regression models for each single factor.
    Uses the same global train/test split as multi-factor models so that
    R² values are directly comparable. Returns a dict of factor -> trained model.
    """
    models = {}
    for factor in factors:
        model = LinearRegression()
        model.fit(X_train[[factor]], y_train)
        models[factor] = model
    return models
 
 
# ── Convenience: Train All Models ────────────────────────────────────────────
 
def train_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """
    Train all four multi-factor regression models and return them as a dictionary.
    Keys: 'linear', 'ridge', 'lasso', 'random_forest'
    """
    return {
        "baseline":      DummyRegressor(strategy="mean").fit(X_train, y_train),
        "linear":        train_linear_regression(X_train, y_train),
        "ridge":         train_ridge(X_train, y_train),
        "lasso":         train_lasso(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
    }