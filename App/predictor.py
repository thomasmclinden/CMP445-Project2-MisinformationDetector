# predictor.py
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from preprocessing import clean_text
import os

MODEL_PATH = "model.pkl"

# --- TRAINING FUNCTION (Run once to create model.pkl) ---
def train_model():
    # Load datasets
    true_df = pd.read_csv("True.csv")
    false_df = pd.read_csv("False.csv")

    # Add labels
    true_df["label"] = "real"
    false_df["label"] = "fake"

    # Use only the 'text' column for training
    df = pd.concat([true_df, false_df], ignore_index=True)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["label"]

    # Simple model pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf", LogisticRegression(max_iter=300))
    ])

    # Train model
    model.fit(X, y)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

# --- LOAD MODEL ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not trained yet. Run train_model().")
    return joblib.load(MODEL_PATH)

model = None
def get_model():
    global model
    if model is None:
        model = load_model()
    return model

# --- PREDICT FOR SCRAPED TEXT ---
def predict_text(text: str) -> str:
    clf = get_model()
    cleaned = clean_text(text)
    pred = clf.predict([cleaned])[0]
    return pred
