# modeling.py

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

def build_classification_pipelines(preprocessor):
    """
    Returns three pipelines (Logistic Regression, Random Forest, XGBoost),
    each combining the given `preprocessor` with a classifier.
    """
    # 1. Logistic Regression pipeline
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

    # 2. Random Forest pipeline
    pipe_rf = Pipeline([
        ('preprocess', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 3. XGBoost pipeline
    #    Note: we omit use_label_encoder since newer XGBoost ignores it
    pipe_xgb = Pipeline([
        ('preprocess', preprocessor),
        ('clf', XGBClassifier(
            objective='multi:softprob',
            num_class=5,
            tree_method='hist',
            eval_metric='mlogloss',
            verbosity=0,        # suppress warnings
            random_state=42,
            n_jobs=-1
        ))
    ])

    return pipe_lr, pipe_rf, pipe_xgb


def tune_model(pipe, param_grid, X_train, y_train, cv_splits=5):
    """
    Performs GridSearchCV for the given pipeline and parameter grid.
    Returns (best_estimator_, best_params, best_score).
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def save_model(model, path: str):
    """
    Saves the given model (pipeline) to disk at `path`.
    """
    joblib.dump(model, path)


def load_model(path: str):
    """
    Loads and returns a pipeline/model from `path`.
    """
    return joblib.load(path)
