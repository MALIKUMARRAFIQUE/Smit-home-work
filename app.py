import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def load_data():
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "DATA" / "telco.csv"

    if not data_path.exists():
        data_path = project_root.parent / "DATA" / "telco.csv"

    df = pd.read_csv(data_path)
    return df


def clean_features(df):
    for col in df.columns:
        values = df[col].astype("string").str.strip().replace("", pd.NA)
        numeric_values = pd.to_numeric(values, errors="coerce")

        # Convert columns that contain only numbers (apart from missing values).
        if numeric_values.notna().sum() == values.notna().sum():
            df[col] = numeric_values.fillna(numeric_values.median())
        else:
            df[col] = values.replace(
                {"<NA>": "Missing", "nan": "Missing", "None": "Missing", "": "Missing"}
            ).fillna("Missing")

    return df


def clean_data(df):
    # target encoding
    target = df["Churn"] if "Churn" in df.columns else None
    if target is None:
        raise ValueError("Input data must contain a 'Churn' column.")

    df = clean_features(df.drop(columns=["Churn"]))
    df["Churn"] = target.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if df["Churn"].isnull().any():
        raise ValueError("Target column contains unexpected values. Check 'Churn' values.")

    return df


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                num_cols
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                cat_cols
            )
        ],
        remainder="drop"
    )

    return preprocessor


def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipe = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    return pipe, metrics


def main():
    df = load_data()
    df = clean_data(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        pipeline, metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append({"Model": name, **metrics})
        trained_models[name] = pipeline
        print(f"{name}: {metrics}")

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print("\nModel Rankings:")
    print(results_df)

    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)

    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, model_dir / filename)

    # Hyperparameter tuning
    rf_pipe = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("model", RandomForestClassifier(random_state=42))
    ])

    rf_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    }

    rf_grid = GridSearchCV(rf_pipe, rf_params, cv=5, scoring="f1", n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    xgb_pipe = Pipeline([
        ("prep", build_preprocessor(X_train)),
        ("model", XGBClassifier(eval_metric="logloss", random_state=42))
    ])

    xgb_params = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.1]
    }

    xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=5, scoring="f1", n_jobs=-1)
    xgb_grid.fit(X_train, y_train)

    tuned_results = []
    for name, grid in [("Tuned Random Forest", rf_grid), ("Tuned XGBoost", xgb_grid)]:
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        tuned_results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob)
        })

    tuned_df = pd.DataFrame(tuned_results)
    final_df = pd.concat([results_df, tuned_df], ignore_index=True).sort_values("F1", ascending=False)
    final_df.to_csv(model_dir / "leaderboard.csv", index=False)
    print("\nFinal leaderboard:")
    print(final_df)

    if rf_grid.best_score_ >= xgb_grid.best_score_:
        champion = rf_grid.best_estimator_
        champion_name = "Tuned Random Forest"
    else:
        champion = xgb_grid.best_estimator_
        champion_name = "Tuned XGBoost"

    joblib.dump(champion, model_dir / "champion_model.pkl")
    print(f"\nChampion: {champion_name}")

    # optional bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(final_df["Model"], final_df["F1"], color="steelblue")
    plt.title("Model F1 Comparison")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()