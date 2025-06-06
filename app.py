# app.py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# --- If you run into SSL issues, skip the downloads and assume data is already present ---
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception:
    pass

model = joblib.load('models/final_csat_LogisticRegression.joblib')

class Ticket(BaseModel):
    Customer_Age: int
    Customer_Gender: str
    Product_Purchased: str
    Ticket_Type: str
    Ticket_Status: str
    Ticket_Priority: str
    Ticket_Channel: str
    Response_Delay: float
    Resolution_Delay: float
    Ticket_Description: str

app = FastAPI(
    title="Customer Satisfaction Prediction API",
    description="POST a support ticket, receive a satisfaction score (1–5).",
    version="1.0"
)

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

@app.get("/")
def read_root():
    return {"message": "Welcome! POST to /predict/ with ticket data to get a satisfaction prediction."}

@app.post("/predict/")
def predict(ticket: Ticket):
    try:
        df = pd.DataFrame([{
            'Customer Age':           ticket.Customer_Age,
            'Customer Gender':        ticket.Customer_Gender,
            'Product Purchased':      ticket.Product_Purchased,
            'Ticket Type':            ticket.Ticket_Type,
            'Ticket Status':          ticket.Ticket_Status,
            'Ticket Priority':        ticket.Ticket_Priority,
            'Ticket Channel':         ticket.Ticket_Channel,
            'Response Delay (hrs)':   ticket.Response_Delay,
            'Resolution Delay (hrs)': ticket.Resolution_Delay,
            'Ticket Description':     ticket.Ticket_Description
        }])

        df['Desc Char Count'] = df['Ticket Description'].str.len()
        df['Desc Word Count'] = df['Ticket Description'].str.split().apply(len)
        df['Cleaned_Desc']   = df['Ticket Description'].apply(clean_text)
        df['Desc Sentiment'] = df['Cleaned_Desc'].apply(lambda x: sia.polarity_scores(x)['compound'])

        pred_shifted = model.predict(df)[0]
        pred_original = int(pred_shifted) + 1
        return {"predicted_satisfaction": pred_original}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Uvicorn will fail here if port 8000 is still in use—be sure you've killed any old process.
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
