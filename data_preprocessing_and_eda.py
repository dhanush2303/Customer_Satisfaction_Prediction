# notebooks/1_data_preprocessing_and_eda.ipynb

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Standard libraries
import os
import re
import numpy as np
import pandas as pd

# 2. Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Text/NLP
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# 4. Preprocessing & Modeling utilities
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# 5. Display settings
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
plt.rcParams['figure.figsize'] = (10, 6)

# 6. Download NLTK data (run once)
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Define path to raw data
DATA_PATH = os.path.join('..', 'data', 'customer_support_tickets.csv')

# Load
df = pd.read_csv("/Users/dhanushadurukatla/Downloads/customer_support_tickets.csv")

# Quick peek
print(f"Number of rows: {len(df)}")
print(df.columns.tolist())
df.head(5)

df.info()
df.isnull().sum().sort_values(ascending=False)

missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
pd.DataFrame({'% Missing': missing})

# Convert to datetime
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')
df['First Response Time'] = pd.to_datetime(df['First Response Time'], errors='coerce')
df['Time to Resolution'] = pd.to_datetime(df['Time to Resolution'], errors='coerce')

# Example: extract year, month
df['Purchase YearMonth'] = df['Date of Purchase'].dt.to_period('M')
df['Ticket Open Hour']    = df['First Response Time'].dt.hour
df['Resolution Hour']     = df['Time to Resolution'].dt.hour

# Compute response and resolution durations (in hours)
df['Response Delay (hrs)'] = (df['First Response Time'] - df['Date of Purchase']).dt.total_seconds() / 3600
df['Resolution Delay (hrs)'] = (df['Time to Resolution'] - df['First Response Time']).dt.total_seconds() / 3600

num_cols = ['Customer Age', 'Response Delay (hrs)', 'Resolution Delay (hrs)', 'Customer Satisfaction Rating']
df[num_cols].describe().T

# Histograms for numeric columns
for col in num_cols:
    plt.figure()
    sns.histplot(data=df, x=col, kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.show()

cat_cols = ['Customer Gender', 'Product Purchased', 'Ticket Type',
            'Ticket Subject', 'Ticket Status', 'Ticket Priority', 'Ticket Channel']

for col in cat_cols:
    top_counts = df[col].value_counts().head(10)
    print(f"\n== {col} ==\n")
    print(top_counts)
    plt.figure()
    sns.barplot(x=top_counts.values, y=top_counts.index, palette='viridis')
    plt.title(f"Top {len(top_counts)} Categories in {col}")
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.show()

# Group by month & count tickets
ticket_trends = df.groupby('Purchase YearMonth').size().sort_index()
plt.figure(figsize=(12, 6))
ticket_trends.plot(marker='o', linestyle='-')
plt.title("Number of Tickets Opened—Month by Month")
plt.xlabel("Year-Month")
plt.ylabel("Ticket Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df_rated = df[df['Customer Satisfaction Rating'].notna()].copy()

plt.figure()
sns.countplot(x='Customer Satisfaction Rating', data=df_rated, palette='coolwarm')
plt.title("Distribution of Satisfaction Ratings (1 to 5)")
plt.xlabel("Satisfaction Rating")
plt.ylabel("Count")
plt.show()

for col in ['Customer Gender', 'Ticket Priority', 'Ticket Channel']:
    summary = df_rated.groupby(col)['Customer Satisfaction Rating'].mean().reset_index()
    plt.figure()
    sns.barplot(x=col, y='Customer Satisfaction Rating', data=summary, palette='magma')
    plt.title(f"Avg. Satisfaction by {col}")
    plt.xlabel(col)
    plt.ylabel("Avg. Satisfaction")
    plt.ylim(1, 5)
    plt.tight_layout()
    plt.show()

plt.figure()
sns.scatterplot(x='Response Delay (hrs)', y='Customer Satisfaction Rating', data=df_rated, alpha=0.3)
plt.title("Satisfaction vs. Response Delay")
plt.xlabel("Response Delay (hours)")
plt.ylabel("Satisfaction Rating")
plt.show()

plt.figure()
sns.scatterplot(x='Resolution Delay (hrs)', y='Customer Satisfaction Rating', data=df_rated, alpha=0.3, color='orange')
plt.title("Satisfaction vs. Resolution Delay")
plt.xlabel("Resolution Delay (hours)")
plt.ylabel("Satisfaction Rating")
plt.show()
