## Project Overview

This repository contains an end-to-end machine learning pipeline and web service for predicting customer satisfaction ratings (1–5) from support ticket data. It demonstrates how to:

1. **Ingest & Clean Data**  
   - Load historical support tickets (CSV) with fields such as customer demographics, ticket metadata, timestamps, and free-text descriptions.  
   - Compute response/resolution delays, filter for valid ratings (1–5), and handle missing values.

2. **Engineer Features**  
   - Numeric: `Customer Age`, `Response Delay (hrs)`, `Resolution Delay (hrs)`.  
   - Categorical: One-hot encode `Gender`, `Product Purchased`, `Ticket Type`, `Status`, `Priority`, `Channel`.  
   - Text: From the ticket description derive  
     - **Char Count**  
     - **Word Count**  
     - **Cleaned Text** (lowercase, punctuation removed, stop-words removed)  
     - **Sentiment Score** (VADER), plus a TF-IDF → SVD embedding.

3. **Train & Select Models**  
   - Build three scikit-learn pipelines (Logistic Regression, Random Forest, XGBoost), each wrapping the same preprocessing.  
   - Perform stratified 5-fold GridSearchCV to tune hyperparameters and compare cross-validation accuracy.  
   - Choose the best model (e.g. XGBoost) and save the trained pipeline as a `.joblib` file.

4. **Deploy as an API**  
   - Use **FastAPI** to expose a `POST /predict/` endpoint.  
   - Clients send a JSON “Ticket” (age, gender, product, delays, description).  
   - The server reconstructs all features, invokes `model.predict()`, and returns a satisfaction score (1–5).

---

## How It Works

1. **Data Loading** (`data_loader.py`)  
   Reads the CSV into a pandas DataFrame.

2. **Preprocessing** (`preprocessing.py`)  
   - Filters out unrated tickets.  
   - Converts date columns to datetime and computes delays in hours.  
   - Drops unused columns and median-imputes numeric features.  
   - Label-encodes low-cardinality categoricals.

3. **Feature Engineering** (`feature_engineering.py`)  
   - Builds a `ColumnTransformer` combining:  
     - Numeric pipeline (impute + scale)  
     - Categorical pipeline (one-hot encoding)  
     - Text pipeline (TF-IDF + Truncated SVD)

4. **Modeling** (`modeling.py`)  
   - Wraps the preprocessor with three classifiers in `Pipeline` objects.  
   - Tunes hyperparameters via `GridSearchCV` on LogisticRegression, RandomForest, and XGBoost.  
   - Returns the best estimator, which is then saved to `models/`.

5. **Evaluation & Saving** (`main.py`)  
   - Splits data into train/test (stratified).  
   - Runs GridSearchCV, selects best model, evaluates on the hold-out test set, and prints accuracy + classification report.  
   - Saves the final pipeline (`.joblib`) and encoders for deployment.

6. **API Service** (`app.py`)  
   - Loads the saved pipeline at startup.  
   - Defines a Pydantic `Ticket` model matching the JSON schema.  
   - On `POST /predict/`, reconstructs all features (including text counts, cleaned text, sentiment), calls `model.predict()`, and returns a JSON with `"predicted_satisfaction": <1–5>`.

---

