import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os

from App.preprocessing import clean_text

MODEL_PATH = "../Models/model2.pkl"


# --- TRAINING FUNCTION ---
def train_model():
    # Load datasets
    print("Loading data...")
    df = pd.read_csv("../Data/Processed/merged_data.csv")
    print(df.head())

    # Clean text
    print("Cleaning text...")
    df["title"] = df["title"].fillna("").apply(clean_text)
    df["text"] = df["text"].fillna("").apply(clean_text)
    df["combined"] = df["title"] + " " + df["text"]

    X = df["combined"]
    y = df["hasMisinformation"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SVM needs calibration for probabilities
    svm_clf = CalibratedClassifierCV(LinearSVC(), cv=5)

    # --- Ensemble with balanced class weights ---
    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("nb", MultinomialNB()),
            ("svm", svm_clf),
            ("sgd", SGDClassifier(loss="log_loss", class_weight="balanced"))
        ],
        voting="soft"
    )

    # --- Pipeline with improved TF-IDF ---
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=15000,
            ngram_range=(1, 3),  # unigrams + bigrams + trigrams
            min_df=2,  # capture rarer but useful terms
        )),
        ("select", SelectKBest(chi2, k="all")),  # keep all features first
        ("clf", ensemble)
    ])

    # Train model
    print("Training model...")
    model.fit(X_train, y_train)

    # --- Correct evaluation ---
    y_pred = model.predict(X_test)
    print(f"\nValidation Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Optional: probability analysis
    y_proba = model.predict_proba(X_test)
    print("Sample probabilities:", y_proba[:5])

    # Save model
    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

# Run once to create model.pkl
# train_model()

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
def predict_text(text: str):
    clf = get_model()
    cleaned = clean_text(text)

    proba = clf.predict_proba([cleaned])[0]  # [prob_real, prob_misinfo]
    pred = int(proba[1] >= 0.5)

    return {
        "prediction": pred,
        "probability_real": float(proba[0]),
        "probability_misinfo": float(proba[1])
    }
