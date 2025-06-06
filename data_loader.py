import pandas as pd
import os

def load_data(data_dir: str, filename: str = '/Users/dhanushadurukatla/Downloads/customer_support_tickets.csv') -> pd.DataFrame:
    """
    Loads the customer support tickets CSV.
    """
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    return df
