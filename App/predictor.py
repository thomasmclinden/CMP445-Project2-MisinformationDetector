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

# -------- FIXED MODEL PATH LOGIC --------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))   # one level up from /App
MODEL_PATH = os.path.join(ROOT_DIR, "Models", "model2.pkl")
# ----------------------------------------


# --- TRAINING FUNCTION ---
def train_model():
    print("Loading data...")
    df = pd.read_csv(os.path.join(ROOT_DIR, "Data", "Processed", "merged_data.csv"))
    print(df.head())

    print("Cleaning text...")
    df["title"] = df["title"].fillna("").apply(clean_text)
    df["text"] = df["text"].fillna("").apply(clean_text)
    df["combined"] = df["title"] + " " + df["text"]

    X = df["combined"]
    y = df["hasMisinformation"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    svm_clf = CalibratedClassifierCV(LinearSVC(), cv=5)

    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ("nb", MultinomialNB()),
            ("svm", svm_clf),
            ("sgd", SGDClassifier(loss="log_loss", class_weight="balanced"))
        ],
        voting="soft"
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=15000,
            ngram_range=(1, 3),
            min_df=2,
        )),
        ("select", SelectKBest(chi2, k="all")),
        ("clf", ensemble)
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\nValidation Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    y_proba = model.predict_proba(X_test)
    print("Sample probabilities:", y_proba[:5])

    joblib.dump(model, MODEL_PATH)
    print("Model saved to", MODEL_PATH)


# --- LOAD MODEL ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not trained yet. Missing: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model


def predict_text(text: str):
    clf = get_model()
    cleaned = clean_text(text)

    proba = clf.predict_proba([cleaned])[0]
    pred = int(proba[1] >= 0.5)

    return {
        "prediction": pred,
        "probability_real": float(proba[0]),
        "probability_misinfo": float(proba[1])
    }
