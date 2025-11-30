# Product Category Classifier  
Machine learning project for predicting the product category based solely on the product title.

This repository contains a complete workflow: data analysis, feature engineering, model comparison, training pipeline, and an interactive script for testing predictions. The project reflects a realistic scenario in e-commerce, where thousands of new products must be categorized quickly and consistently.

---

## 📂 Project Structure

```
ml-product-category-classifier/
├── data/
│   └── products.csv
├── notebook/
│   └── product_category_analysis.ipynb
├── results/
│   └── model_comparison.txt
├── models/                          # (generated locally after training)
│   └── product_category_model.pkl
├── src/
│   ├── train_model.py
│   └── predict_category.py
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The `models/` folder is created locally when you run `train_model.py` and is excluded from Git (listed in `.gitignore`).

---

## 📘 Project Description

The goal is to build a machine learning model that predicts the correct product category using only the product title.  
This solves a practical problem in online retail: manual classification is slow, inconsistent, and hard to scale.

The model uses:

- TF-IDF vectorization of product titles  
- Additional numeric features (length, word count, digit indicators, etc.)  
- Multiple classification algorithms for comparison  
- Final model stored as a reusable pipeline in `.pkl` format  

The notebook contains the full analysis and model selection process, while `train_model.py` and `predict_category.py` allow the project to run independently from the notebook.

---

## 🧠 Key Features

- Complete EDA and data cleaning  
- Custom feature engineering on product titles  
- ColumnTransformer with TF-IDF + scaled numeric features  
- Comparison of Logistic Regression, Linear SVC, and Random Forest  
- Final model exported as `product_category_model.pkl`  
- Interactive CLI prediction script  
- Transparent model evaluation stored in `results/model_comparison.txt`

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/DennisForge/ml-product-category-classifier.git
cd ml-product-category-classifier
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Jupyter Notebook

To view the full analysis:

```bash
jupyter notebook notebook/product_category_analysis.ipynb
```

The notebook includes:

- Data exploration  
- Cleaning and preprocessing  
- Feature engineering  
- Training and evaluating multiple models  
- Confusion matrix visualization  
- Exporting the final pipeline  

---

## 🏋️ Training the Model

Run the training script:

```bash
python src/train_model.py
```

This will:

- Load and preprocess the dataset  
- Train all models  
- Save the final model to `models/product_category_model.pkl`  
- Save accuracy comparison to `results/model_comparison.txt`  

---

## 🔍 Predicting Categories

Use the interactive prediction script:

```bash
python src/predict_category.py
```

You will be prompted to enter any product title, for example:

```
Enter product title: bosch serie 4 kgv39vl31g
Predicted category: Fridge Freezers
```

---

## 📦 Dataset

The project uses `products.csv`, which contains:

- Product ID  
- Product Title  
- Merchant ID  
- Category Label  
- Product Code  
- Number of Views  
- Merchant Rating  
- Listing Date  

All analysis and training rely solely on this dataset.

---

## 📝 Notes

- The final model is trained on the full cleaned dataset.  
- The entire preprocessing pipeline is stored inside the `.pkl` file.  
- Both scripts (`train_model.py` and `predict_category.py`) can run independently of the notebook.
- The `models/` folder is not tracked in Git to keep the repository clean - it will be created automatically when you train the model.

---

## 👤 Author

**Created by:** [DennisForge](https://github.com/DennisForge)

> *"Day one, not someday. Build, learn, ship."*

---
