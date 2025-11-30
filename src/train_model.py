"""
Treniranje i cuvanje modela za klasifikaciju proizvoda po kategorijama
na osnovu naslova (Product Title).

Koraci:
- ucitavanje i ciscenje podataka iz data/products.csv
- feature engineering za naslov proizvoda
- treniranje vise modela (LogisticRegression, LinearSVC, RandomForest)
- izbor najboljeg modela prema accuracy na test skupu
- treniranje finalnog pipeline-a na celom skupu
- cuvanje pipeline-a u models/product_category_model.pkl
- cuvanje rezultata poređenja modela u results/model_comparison.txt
"""

import os
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

import joblib


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw_data(root_dir: str) -> pd.DataFrame:
    """UUcitaj originalni CSV fajl sa proizvodima."""
    data_path = os.path.join(root_dir, "data", "products.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Nije pronadjen fajl: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"[INFO] Ucitani podaci: {df_raw.shape[0]} redova, {df_raw.shape[1]} kolona")
    return df_raw


def clean_and_prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Ciscenje i priprema podataka:
    - standardizacija naziva kolona
    - uklanjanje redova bez naslova ili kategorije
    - popunjavanje numerickih kolona median vrednostima
    - obradjivanje datuma
    - feature engineering za naslov proizvoda
    """
    df = df_raw.copy()

    # 1) standardizacija naziva kolona
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns=lambda c: c.lower().replace(" ", "_"))

    if "_product_code" in df.columns:
        df = df.rename(columns={"_product_code": "product_code"})

    # 2) uklanjanje redova bez naslova ili kategorije
    if "product_title" not in df.columns or "category_label" not in df.columns:
        raise KeyError("Ocekivane kolone 'product_title' i/ili 'category_label' ne postoje u skupu podataka.")

    before_rows = df.shape[0]
    df = df.dropna(subset=["product_title", "category_label"])
    after_rows = df.shape[0]
    print(f"[INFO] Uklonjeno redova bez naslova ili kategorije: {before_rows - after_rows}")
    print(f"[INFO] Nove dimenzije skupa: {df.shape}")

    # 3) popunjavanje numerickih kolona median vrednostima
    if "number_of_views" in df.columns:
        df["number_of_views"] = df["number_of_views"].fillna(df["number_of_views"].median())

    if "merchant_rating" in df.columns:
        df["merchant_rating"] = df["merchant_rating"].fillna(df["merchant_rating"].median())

    # 4) listing_date -> datetime + popuna najcescim datumom
    if "listing_date" in df.columns:
        df["listing_date"] = pd.to_datetime(df["listing_date"], errors="coerce")
        if df["listing_date"].isna().sum() > 0:
            most_common_date = df["listing_date"].mode()[0]
            df["listing_date"] = df["listing_date"].fillna(most_common_date)

    # 5) feature engineering za naslov
    df["title_char_len"] = df["product_title"].str.len()
    df["title_word_count"] = df["product_title"].str.split().str.len()
    df["title_digit_count"] = df["product_title"].str.count(r"\d")
    df["has_digits"] = (df["title_digit_count"] > 0).astype(int)
    df["has_upper_acronym"] = df["product_title"].str.contains(r"\b[A-Z]{2,}\b", regex=True).astype(int)

    return df


def build_feature_sets(df: pd.DataFrame):
    """Definisi X i y kao u notebook-u."""
    text_feature = "product_title"

    numeric_features: List[str] = [
        "title_char_len",
        "title_word_count",
        "title_digit_count",
        "has_digits",
        "has_upper_acronym",
    ]

    for col in ["number_of_views", "merchant_rating"]:
        if col in df.columns:
            numeric_features.append(col)

    X = df[[text_feature] + numeric_features].copy()
    y = df["category_label"].copy()

    return X, y, text_feature, numeric_features


def build_preprocessor(text_feature: str, numeric_features: List[str]) -> ColumnTransformer:
    """Izgradi ColumnTransformer za tekst i numericke feature-e."""
    text_transformer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_features=50000,
    )

    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, text_feature),
            ("num", numeric_transformer, numeric_features),
        ]
    )
    return preprocessor


def get_models() -> Dict[str, Any]:
    """Vrati recnik modela koje testiramo."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=200,
            n_jobs=-1,
            multi_class="multinomial",
        ),
        "LinearSVC": LinearSVC(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }


def main() -> None:
    root_dir = get_project_root()
    print(f"[INFO] Root folder projekta: {root_dir}")

    # 1) učitaj i očisti podatke
    df_raw = load_raw_data(root_dir)
    df = clean_and_prepare(df_raw)

    # 2) definiši X, y i feature liste
    X, y, text_feature, numeric_features = build_feature_sets(df)
    print(f"[INFO] X shape: {X.shape}, y length: {len(y)}")

    # 3) train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"[INFO] Trening skup: {X_train.shape}, Test skup: {X_test.shape}")

    # 4) preprocesor i modeli
    preprocessor = build_preprocessor(text_feature, numeric_features)
    models = get_models()

    results = []
    trained_pipelines: Dict[str, Pipeline] = {}

    # 5) treniranje i poredjenje modela
    for name, clf in models.items():
        print("=" * 80)
        print(f"[INFO] Treniranje modela: {name}")
        print("=" * 80)

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"[RESULT] Accuracy na test skupu ({name}): {acc:.4f}\n")
        print("Classification report:\n")
        print(classification_report(y_test, y_pred))

        results.append({"model": name, "accuracy": acc})
        trained_pipelines[name] = pipe

    # 6) izbor najboljeg modela
    results_df = pd.DataFrame(results).sort_values(by="accuracy", ascending=False)
    best_row = results_df.iloc[0]
    best_model_name = best_row["model"]
    best_accuracy = best_row["accuracy"]

    print("=" * 80)
    print(f"[BEST] Najbolji model: {best_model_name} (accuracy = {best_accuracy:.4f})")
    print("=" * 80)

    # 7) treniranje finalnog pipeline-a na celom skupu
    final_clf = models[best_model_name]
    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", final_clf),
    ])

    print("[INFO] Treniranje finalnog pipeline-a na celom skupu...")
    final_pipeline.fit(X, y)

    # 8) cuvanje modela
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "product_category_model.pkl")

    joblib.dump(final_pipeline, model_path)
    print(f"[SAVE] Finalni pipeline sačuvan u: {model_path}")

    # 9) cuvanje poredjenja modela
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    comparison_path = os.path.join(results_dir, "model_comparison.txt")

    with open(comparison_path, "w") as f:
        f.write("Model comparison (accuracy on test set)\n")
        f.write("-" * 50 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['model']}: {row['accuracy']:.4f}\n")

    print(f"[SAVE] Rezultati poredjenja modela sacuvani u: {comparison_path}")
    print("[DONE] Treniranje zavrseno uspesno.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
