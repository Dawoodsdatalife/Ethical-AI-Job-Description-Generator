import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import streamlit as st
import joblib

# Download NLTK data if not already downloaded
nltk.download('punkt')

# Model Options
MODEL_OPTIONS = {
    'Logistic Regression': LogisticRegression(max_iter=300),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Naive Bayes': MultinomialNB(),
    'Support Vector Machine': SVC(),
    'Neural Network': MLPClassifier(max_iter=300)
}

def preprocess_text(text):
    """Preprocess textual data."""
    if isinstance(text, str):
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join(tokens)
    return ''

def preprocess_data(data):
    """Prepare and preprocess the dataset."""
    # Ensure required columns exist
    required_columns = ['Description', 'Label', 'Gender', 'YearsCoding', 'EmploymentStatus']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Dataset must include {', '.join(required_columns)}.")

    # Preprocess text descriptions
    data['Processed_Description'] = data['Description'].apply(preprocess_text)

    # Encode categorical variables
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x.lower() == 'male' else 0)  # Example binary encoding
    data['EmploymentStatus'] = pd.Categorical(data['EmploymentStatus']).codes

    # Prepare the input (X) and output (y)
    X = data[['Processed_Description', 'Gender', 'YearsCoding', 'EmploymentStatus']]
    y = data['Label'].apply(lambda x: 1 if x == 'Inclusive' else 0)

    # Split into training and testing datasets
    return train_test_split(X, y, test_size=0.2, random_state=42)

def vectorize_text(X_train, X_test):
    """Vectorize textual data and combine with numerical features."""
    # Separate textual data for vectorization
    X_train_text = X_train['Processed_Description']
    X_test_text = X_test['Processed_Description']
    
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Combine vectorized text with other numerical features
    X_train_combined = np.hstack((X_train_tfidf.toarray(), X_train[['Gender', 'YearsCoding', 'EmploymentStatus']].to_numpy()))
    X_test_combined = np.hstack((X_test_tfidf.toarray(), X_test[['Gender', 'YearsCoding', 'EmploymentStatus']].to_numpy()))

    return X_train_combined, X_test_combined, vectorizer

def train_model(X_train, y_train, model_choice):
    """Train the selected model."""
    model = MODEL_OPTIONS[model_choice]
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

def main():
    """Main function to run Streamlit app."""
    st.title("Ethical AI Job Description Generator")

    # Sidebar for user input
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox("Choose a Model", list(MODEL_OPTIONS.keys()))
    file_path = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    if file_path is not None:
        try:
            # Load and preprocess data
            data = pd.read_csv(file_path)
            st.write("Dataset Preview:", data.head())  # Display dataset preview
            X_train, X_test, y_train, y_test = preprocess_data(data)
            X_train_combined, X_test_combined, vectorizer = vectorize_text(X_train, X_test)

            # Train and evaluate the model
            model = train_model(X_train_combined, y_train, model_choice)
            accuracy, report = evaluate_model(model, X_test_combined, y_test)

            # Display results
            st.write(f"### Model: {model_choice}")
            st.write(f"Accuracy: {accuracy:.4f}")
            st.text("Classification Report:")
            st.text(report)

            # Save trained model (optional)
            joblib.dump(model, "trained_model.pkl")
            st.success("Model trained and saved successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
