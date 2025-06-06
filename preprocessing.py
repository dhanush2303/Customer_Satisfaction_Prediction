import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def preprocess_df(df: pd.DataFrame, target: str = 'Customer Satisfaction Rating'):
    """
    Takes raw DataFrame and returns (X, y) ready for feature engineering/modeling.
    """
    # 1. Drop unrated rows
    df = df[df[target].notna()].copy()

    # 2. Convert date columns
    df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
    df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
    df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')

    # 3. Compute delays
    df['Response Delay (hrs)'] = (df['First Response Time'] - df['Date of Purchase']).dt.total_seconds() / 3600
    df['Resolution Delay (hrs)'] = (df['Time to Resolution'] - df['First Response Time']).dt.total_seconds() / 3600

    # 4. Drop unused columns
    drop_cols = ['Ticket ID', 'Customer Name', 'Customer Email', 'Resolution',
                 'Ticket Subject', 'Date of Purchase', 'First Response Time', 'Time to Resolution']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # 5. Handle missing numeric values
    num_cols = ['Customer Age', 'Response Delay (hrs)', 'Resolution Delay (hrs)']
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # 6. Label encode low-cardinality categorical
    low_card_cols = ['Customer Gender', 'Product Purchased', 'Ticket Type',
                     'Ticket Status', 'Ticket Priority', 'Ticket Channel']
    label_encoders = {}
    for col in low_card_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # 7. Text cleaning & sentiment
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.sentiment import SentimentIntensityAnalyzer
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

    stop_words = set(stopwords.words('english'))
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words]
        return " ".join(tokens)

    df['Desc Char Count'] = df['Ticket Description'].str.len()
    df['Desc Word Count'] = df['Ticket Description'].str.split().apply(len)
    df['Cleaned_Desc'] = df['Ticket Description'].apply(clean_text)

    sia = SentimentIntensityAnalyzer()
    df['Desc Sentiment'] = df['Cleaned_Desc'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # 8. Define features & target
    X = df.drop(columns=[target, 'SatisfiedFlag'], errors='ignore')
    # Shift ratings 1–5 → 0–4 for all downstream models
    y = df[target].astype(int) - 1

    return X, y, label_encoders
