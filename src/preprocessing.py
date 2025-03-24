import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

def prepare_data(data):
    data['Processed_Description'] = data['Description'].apply(preprocess_text)
    X = data['Processed_Description']
    y = data['Label'].apply(lambda x: 1 if x == 'Inclusive' else 0)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer