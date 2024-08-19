import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_text(text):
    # Add text preprocessing steps here (e.g., lowercasing, removing punctuation)
    return text.lower()

def preprocess_data(df):
    df['review'] = df['review'].apply(preprocess_text)
    return df

def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['review'], df['sentiment'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
