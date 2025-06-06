# feature_engineering_and_modeling.py

########################################
# 1. SSL workaround + imports
########################################
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib

sns.set(style="whitegrid")
pd.set_option('display.max_columns', 100)
plt.rcParams['figure.figsize'] = (10, 6)

# Download NLTK data (stopwords & VADER)
nltk.download('stopwords')
nltk.download('vader_lexicon')


########################################
# 2. Load CSV
########################################
DATA_PATH = '/Users/dhanushadurukatla/Downloads/customer_support_tickets.csv'
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"Could not find {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)
print(df.columns.tolist())


########################################
# 3. Convert datetimes and compute delays
########################################
df['Date of Purchase']     = pd.to_datetime(df['Date of Purchase'], errors='coerce')
df['First Response Time']  = pd.to_datetime(df['First Response Time'], errors='coerce')
df['Time to Resolution']   = pd.to_datetime(df['Time to Resolution'], errors='coerce')

df['Response Delay (hrs)']   = (
    df['First Response Time'] - df['Date of Purchase']
).dt.total_seconds() / 3600

df['Resolution Delay (hrs)'] = (
    df['Time to Resolution'] - df['First Response Time']
).dt.total_seconds() / 3600

# Now drop the raw datetime columns
df.drop(columns=[
    'Date of Purchase',
    'First Response Time',
    'Time to Resolution'
], inplace=True)

print("After computing delays:", df.shape)
assert 'Response Delay (hrs)' in df.columns
assert 'Resolution Delay (hrs)' in df.columns


########################################
# 4. Filter to only rated tickets & drop unneeded cols
########################################
df = df[df['Customer Satisfaction Rating'].notna()].copy()
print("After filtering rated tickets:", df.shape)

to_drop = [
    'Ticket ID',
    'Customer Name',
    'Customer Email',
    'Resolution',
    'Ticket Subject'
]
df.drop(columns=to_drop, inplace=True, errors='ignore')
print("After dropping unneeded columns:", df.shape)


########################################
# 5. Impute numeric columns
########################################
numeric_cols = ['Customer Age', 'Response Delay (hrs)', 'Resolution Delay (hrs)']
num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])


########################################
# 6. Label‐encode categorical columns
########################################
categorical_cols = [
    'Customer Gender',
    'Product Purchased',
    'Ticket Type',
    'Ticket Status',
    'Ticket Priority',
    'Ticket Channel'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le


########################################
# 7. Text features: description cleaning & sentiment
########################################
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df['Desc Char Count'] = df['Ticket Description'].str.len()
df['Desc Word Count'] = df['Ticket Description'].str.split().apply(len)
df['Cleaned_Desc']   = df['Ticket Description'].apply(clean_text)

sia = SentimentIntensityAnalyzer()
df['Desc Sentiment'] = df['Cleaned_Desc'].apply(lambda x: sia.polarity_scores(x)['compound'])


########################################
# 8. Prepare X, y, and train/test split
########################################
target_col = 'Customer Satisfaction Rating'
feature_cols = (
    numeric_cols +
    ['Desc Char Count', 'Desc Word Count', 'Desc Sentiment'] +
    categorical_cols +
    ['Cleaned_Desc']
)

X = df[feature_cols].copy()

# ==== SHIFT TARGET: 1–5 → 0–4 ====
y = df[target_col].astype(int) - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Train class dist:\n", y_train.value_counts(normalize=True))
print("Test class dist:\n", y_test.value_counts(normalize=True))


########################################
# 9. Build preprocessing & modeling pipelines
########################################
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

n_svd_components = 100
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        stop_words='english'
    )),
    ('svd', TruncatedSVD(n_components=n_svd_components, random_state=42))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, categorical_cols),
    ('txt', text_pipeline, 'Cleaned_Desc'),
], remainder='drop')

pipe_lr = Pipeline([
    ('preprocess', preprocessor),
    ('clf', LogisticRegression(
        penalty='l2',
        solver='saga',
        multi_class='multinomial',
        class_weight='balanced',
        max_iter=500,
        random_state=42
    ))
])

pipe_rf = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

pipe_xgb = Pipeline([
    ('preprocess', preprocessor),
    ('clf', XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        tree_method='hist',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ))
])


########################################
# 10. Hyperparameter grids & CV
########################################
param_grid_lr = {
    'clf__C': [0.01, 0.1, 1]
}
param_grid_rf = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 10],
    'clf__min_samples_leaf': [1, 2]
}
param_grid_xgb = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 5],
    'clf__learning_rate': [0.05, 0.1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nTuning Logistic Regression...")
grid_lr = GridSearchCV(
    estimator=pipe_lr,
    param_grid=param_grid_lr,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)
grid_lr.fit(X_train, y_train)
print(" Best LR params:", grid_lr.best_params_)
print(" Best LR CV accuracy:", grid_lr.best_score_)

print("\nTuning Random Forest...")
grid_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)
grid_rf.fit(X_train, y_train)
print(" Best RF params:", grid_rf.best_params_)
print(" Best RF CV accuracy:", grid_rf.best_score_)

print("\nTuning XGBoost...")
grid_xgb = GridSearchCV(
    estimator=pipe_xgb,
    param_grid=param_grid_xgb,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    verbose=2
)
grid_xgb.fit(X_train, y_train)
print(" Best XGB params:", grid_xgb.best_params_)
print(" Best XGB CV accuracy:", grid_xgb.best_score_)


########################################
# 11. Select best model & evaluate on test set
########################################
cv_scores = {
    'LogisticRegression': grid_lr.best_score_,
    'RandomForest': grid_rf.best_score_,
    'XGBoost': grid_xgb.best_score_
}
print("\nCV Scores:", cv_scores)

best_name = max(cv_scores, key=cv_scores.get)
print("Best model by CV:", best_name)

if best_name == 'LogisticRegression':
    best_model = grid_lr.best_estimator_
elif best_name == 'RandomForest':
    best_model = grid_rf.best_estimator_
else:
    best_model = grid_xgb.best_estimator_

print(f"\nEvaluating {best_name} on test set...")
y_pred = best_model.predict(X_test)

# ==== SHIFT PREDICTIONS BACK FOR METRICS (0–4 → 1–5) ====
y_test_display = y_test + 1
y_pred_display = y_pred + 1

test_acc = accuracy_score(y_test, y_pred)
print(" Test Accuracy:", test_acc)
print("\n Classification Report (1–5 labels):\n", classification_report(y_test_display, y_pred_display))


cf_fname = f"confusion_matrix_{best_name}.png"
cm = confusion_matrix(y_test_display, y_pred_display, labels=[1,2,3,4,5])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[1,2,3,4,5],
            yticklabels=[1,2,3,4,5])
plt.title(f"Confusion Matrix ({best_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(cf_fname)
plt.show()
print(f" Confusion matrix saved to {cf_fname}")


########################################
# 12. If RF or XGB, plot feature importances
########################################
if best_name in ['RandomForest', 'XGBoost']:
    print(f"\nPlotting feature importances for {best_name}...")
    column_transformer = best_model.named_steps['preprocess']
    feat_names = []

    # Numeric names
    feat_names.extend(numeric_cols)

    # Categorical one-hot names
    cat_transformer = column_transformer.named_transformers_['cat']
    ohe = cat_transformer.named_steps['onehot']
    cat_encoded = ohe.get_feature_names_out(categorical_cols)
    feat_names.extend(cat_encoded)

    # Text SVD names
    svd_names = [f"svd_{i}" for i in range(n_svd_components)]
    feat_names.extend(svd_names)

    importances = best_model.named_steps['clf'].feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': feat_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    top20 = feat_imp_df.head(20)
    plt.figure(figsize=(8, 8))
    sns.barplot(x='importance', y='feature', data=top20, palette='viridis')
    plt.title(f"Top 20 Importances ({best_name})")
    plt.tight_layout()
    fi_fname = f"feature_importances_{best_name}.png"
    plt.savefig(fi_fname)
    plt.show()
    print(f" Feature importance plot saved to {fi_fname}")


########################################
# 13. Save best model + encoders
########################################
model_fname = f"best_customer_satisfaction_model_{best_name}.joblib"
joblib.dump(best_model, model_fname)
print(f"\nSaved best model to {model_fname}")

encoders_fname = "label_encoders.joblib"
joblib.dump(label_encoders, encoders_fname)
print(f"Saved label encoders to {encoders_fname}")
