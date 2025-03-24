import pandas as pd
import nltk
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
import torch
import joblib
from alibi.explainers import AnchorText
import streamlit as st

nltk.download('punkt')

# Load dataset from CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        tokens = nltk.word_tokenize(text.lower())
        return ' '.join(tokens)
    else:
        return ''

# Preparing the data
def prepare_data(data):
    if 'Description' not in data.columns:
        raise ValueError('The column "Description" is missing from the CSV file.')
    
    data['Processed_Description'] = data['Description'].apply(preprocess_text)
    X = data['Processed_Description']
    y = data['Label'].apply(lambda x: 1 if x == 'Inclusive' else 0)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorizing text data
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, vectorizer

# Train and evaluate models
def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

# Save the best model
def save_model(model, filename):
    joblib.dump(model, filename)

# Load a saved model
def load_model(filename):
    return joblib.load(filename)

# Interpret model predictions with AnchorText
def interpret_with_anchor_text(model, vectorizer, X_train):
    explainer = AnchorText()
    sample_text = [X_train.iloc[0]]  # Using a single text sample from the training data
    prediction_fn = lambda texts: model.predict(vectorizer.transform(texts))  # Prediction function
    
    explanation = explainer.explain(sample_text[0], prediction_fn)
    print('Explanation:', explanation)
    return 'Anchor Text Interpretation Completed'

# Evaluate fairness of the model
def evaluate_fairness(X, y, model):
    X_array = X.toarray()  # Convert sparse matrix to numpy array

    # Create a DataFrame from X_array for processing
    X_df = pd.DataFrame(X_array)

    # Concatenate X and y to form a single DataFrame
    df = pd.concat([X_df, pd.Series(y, name='Label')], axis=1)
    
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Reassign X and y after cleaning the data
    X_array = df.iloc[:, :-1].values  # All columns except the last one
    y_clean = df['Label'].values  # Only the last column
    
    # Create the BinaryLabelDataset
    dataset = BinaryLabelDataset(
        favorable_label=1, 
        unfavorable_label=0,
        df=df,
        label_names=['Label'],
        protected_attribute_names=['Processed_Description']
    )
    
    metric = BinaryLabelDatasetMetric(
        dataset, 
        privileged_groups=[{'Processed_Description': 1}],
        unprivileged_groups=[{'Processed_Description': 0}]
    )
    
    classification_metric = ClassificationMetric(
        dataset, 
        dataset, 
        model.predict(X_array),
        privileged_groups=[{'Processed_Description': 1}],
        unprivileged_groups=[{'Processed_Description': 0}]
    )
    
    print(f"\nStatistical Parity Difference: {metric.mean_difference()}")
    print(f"Equal Opportunity Difference: {classification_metric.equal_opportunity_difference()}")
    print(f"Disparate Impact: {metric.disparate_impact()}")
    
    return (metric.mean_difference(), 
            classification_metric.equal_opportunity_difference(), 
            metric.disparate_impact())


    # Define models to test
    models = [
        LogisticRegression(max_iter=300),
        DecisionTreeClassifier(max_depth=12),
        RandomForestClassifier(n_estimators=150),
        MultinomialNB(),
        SVC(kernel='linear', C=1.0),
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300)
    ]

    best_model = None
    best_accuracy = 0

    # Train and evaluate all models
    for model in models:
        trained_model, accuracy = train_and_evaluate_model(X_train_tfidf, X_test_tfidf, y_train, y_test, model)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = trained_model

    # Save the best performing model
    save_model(best_model, 'best_model.pkl')

    # Evaluate fairness
    evaluate_fairness(X_train_tfidf, y_train, best_model)

    # Interpret predictions
    interpret_with_anchor_text(best_model, vectorizer, X_train)
