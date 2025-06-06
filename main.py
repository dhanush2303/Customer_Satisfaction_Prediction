# main.py

import os
import ssl
from data_loader import load_data
from preprocessing import preprocess_df
from feature_engineering import build_preprocessor
from modeling import build_classification_pipelines, tune_model, save_model
from utils import plot_confusion_matrix, print_classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# SSL workaround (if needed by NLTK in other modules)
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    # 1. Load raw data
    print("Loading data...")
    df = load_data(
        data_dir='',
        filename='/Users/dhanushadurukatla/Downloads/customer_support_tickets.csv'
    )

    # 2. Preprocess and split
    print("Preprocessing data...")
    X, y, label_encoders = preprocess_df(df, target='Customer Satisfaction Rating')

    # Debug: inspect unique values before cleaning
    unique_vals = sorted(y.unique())
    print("Unique raw ratings (before cleaning):", unique_vals)

    # Filter to keep only ratings 1..5 (drop 0 or any out‐of‐range entries)
    valid_mask = y.between(1, 5)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # Shift from [1..5] → [0..4]
    y = y.astype(int) - 1

    # Debug: verify new unique range after shift
    print("Unique cleaned labels (after shift):", sorted(y.unique()))

    # 3. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 4. Build preprocessor & pipelines
    print("\nBuilding preprocessing & model pipelines...")
    preprocessor = build_preprocessor()
    pipe_lr, pipe_rf, pipe_xgb = build_classification_pipelines(preprocessor)

    # 5. Hyperparameter grids
    param_grid_lr = {
        'clf__C': [0.01, 0.1, 1],
    }
    param_grid_rf = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10],
    }
    param_grid_xgb = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.05, 0.1],
    }

    # 6. Tune each model
    print("\nTuning Logistic Regression...")
    best_lr, lr_params, lr_score = tune_model(pipe_lr, param_grid_lr, X_train, y_train)
    print("  LR Params:", lr_params, "CV Score:", lr_score)

    print("\nTuning Random Forest...")
    best_rf, rf_params, rf_score = tune_model(pipe_rf, param_grid_rf, X_train, y_train)
    print("  RF Params:", rf_params, "CV Score:", rf_score)

    print("\nTuning XGBoost...")
    best_xgb, xgb_params, xgb_score = tune_model(pipe_xgb, param_grid_xgb, X_train, y_train)
    print("  XGB Params:", xgb_params, "CV Score:", xgb_score)

    # 7. Select best model by CV
    cv_scores = {
        'LogisticRegression': lr_score,
        'RandomForest': rf_score,
        'XGBoost': xgb_score
    }
    print("\nCV Scores:", cv_scores)

    best_name = max(cv_scores, key=cv_scores.get)
    print("Best model by CV:", best_name)

    if best_name == 'LogisticRegression':
        best_model = best_lr
    elif best_name == 'RandomForest':
        best_model = best_rf
    else:
        best_model = best_xgb

    # 8. Evaluate on test set
    print(f"\nEvaluating {best_name} on test set...")
    y_pred = best_model.predict(X_test)

    # Shift back up to 1–5 for readability
    y_test_display = y_test + 1
    y_pred_display = y_pred + 1

    print("\nClassification Report (labels 1–5):")
    print(classification_report(y_test_display, y_pred_display))

    # Plot confusion matrix with classes [1,2,3,4,5]
    plot_confusion_matrix(
        y_true=y_test_display,
        y_pred=y_pred_display,
        classes=[1, 2, 3, 4, 5]
    )

    # 9. Save the best model
    os.makedirs('models', exist_ok=True)
    # Always save the chosen model under a fixed filename (e.g. RF), so app.py can load it.
    # If RF was not the best, overwrite its file with this best_model.
    model_fname = 'models/final_csat_rf.joblib'
    save_model(best_model, model_fname)
    print(f"Saved best model to {model_fname}")

    # 10. (Optional) Save label encoders
    import joblib
    joblib.dump(label_encoders, 'models/label_encoders.joblib')

if __name__ == "__main__":
    main()

