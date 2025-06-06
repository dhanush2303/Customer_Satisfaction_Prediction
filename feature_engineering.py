# feature_engineering.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def build_preprocessor():
    numeric_cols = [
        'Customer Age', 'Response Delay (hrs)', 'Resolution Delay (hrs)',
        'Desc Char Count', 'Desc Word Count', 'Desc Sentiment'
    ]
    categorical_cols = [
        'Customer Gender', 'Product Purchased', 'Ticket Type',
        'Ticket Status', 'Ticket Priority', 'Ticket Channel'
    ]
    text_col = 'Cleaned_Desc'

    # Numeric pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline (use sparse_output=False instead of sparse=False)
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Text pipeline
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.85,
            stop_words='english'
        )),
        ('svd', TruncatedSVD(n_components=100, random_state=42))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols),
        ('txt', text_pipeline, text_col),
    ], remainder='drop')

    return preprocessor
