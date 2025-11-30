"""
Interaktivno testiranje treniranog modela za klasifikaciju proizvoda po
kategorijama na osnovu naslova (Product Title).

Koriscenje:
    python src/predict_category.py
    -> unos naslova proizvoda u terminal
"""

import os
import sys
from typing import List

import pandas as pd
import joblib


def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_feature_row(title: str) -> pd.DataFrame:
    df = pd.DataFrame({"product_title": [title]})

    # isti feature engineering kao u treningu
    df["title_char_len"] = df["product_title"].str.len()
    df["title_word_count"] = df["product_title"].str.split().str.len()
    df["title_digit_count"] = df["product_title"].str.count(r"\d")
    df["has_digits"] = (df["title_digit_count"] > 0).astype(int)
    df["has_upper_acronym"] = df["product_title"].str.contains(r"\b[A-Z]{2,}\b", regex=True).astype(int)

    # numericke kolone koje model ocekuje (number_of_views, merchant_rating)
    # postavljamo na 0.0 kao neutralnu vrednost
    df["number_of_views"] = 0.0
    df["merchant_rating"] = 0.0

    # redosled kolona mora da odgovara onome iz treninga
    text_feature = "product_title"
    numeric_features: List[str] = [
        "title_char_len",
        "title_word_count",
        "title_digit_count",
        "has_digits",
        "has_upper_acronym",
        "number_of_views",
        "merchant_rating",
    ]

    return df[[text_feature] + numeric_features]


def load_model(root_dir: str):
    """UUcitaj sacuvani pipeline iz models/product_category_model.pkl."""
    model_path = os.path.join(root_dir, "models", "product_category_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nije pronadjen model na lokaciji: {model_path}\n"
                                f"Prvo pokreni: python src/train_model.py")
    pipeline = joblib.load(model_path)
    return pipeline


def main() -> None:
    root_dir = get_project_root()
    print(f"[INFO] Root folder projekta: {root_dir}")

    try:
        pipeline = load_model(root_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Product Category Predictor ===")
    print("Unesite naziv proizvoda za koji zelite predikciju kategorije.")
    print("Upisite 'exit' ili 'q' za izlaz.\n")

    examples = [
        "iphone 7 32gb gold,4,3,Apple iPhone 7 32GB",
        "olympus e m10 mark iii geh use silber",
        "kenwood k20mss15 solo",
        "bosch wap28390gb 8kg 1400 spin",
        "bosch serie 4 kgv39vl31g",
        "smeg sbs8004po",
    ]
    print("Primeri naziva koje mozete probati:")
    for ex in examples:
        print(f"  - {ex}")
    print()

    while True:
        try:
            user_input = input("\nUnesite naziv proizvoda: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Izlaz iz programa.")
            break

        if not user_input:
            print("Unos je prazan, pokusajte ponovo.")
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("[INFO] Zavrsetak rada.")
            break

        features_df = build_feature_row(user_input)
        pred = pipeline.predict(features_df)[0]

        print(f"Predvidjena kategorija: {pred}")


if __name__ == "__main__":
    main()
