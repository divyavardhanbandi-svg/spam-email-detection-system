"""
Spam Email Detection System
---------------------------
Pipeline: load data -> clean text -> TF-IDF -> Naive Bayes / Logistic Regression
          -> evaluate -> save model -> predict new emails.

Usage:
    python spam_detector.py                      # train on built-in sample data
    python spam_detector.py --data spam.csv      # train on your own CSV
    python spam_detector.py --predict "You won a free prize, click now!"

CSV format: two columns named  label,text   (label = spam / ham)
A good free dataset: "SMS Spam Collection" or "Enron Spam" from Kaggle / UCI.
"""

import argparse
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

MODEL_PATH = "spam_model.joblib"

# ---------------------------------------------------------------- sample data
SAMPLE_SPAM = [
    "Congratulations! You have won a free iPhone. Click here to claim now",
    "URGENT: Your bank account is suspended. Verify your password immediately",
    "Get cheap meds online, no prescription needed, limited time offer",
    "You are selected for a $1000 gift card. Reply YES to claim your prize",
    "Make money fast working from home. Earn $5000 per week guaranteed",
    "Free entry in our weekly lottery draw. Text WIN to 80085 now",
    "Final notice: your payment is overdue. Pay now to avoid legal action",
    "Exclusive deal!!! 90% off luxury watches. Buy now before stock ends",
    "Dear customer, your account has been compromised. Login to secure it",
    "Win a brand new car! Click the link to enter the giveaway today",
    "Lowest mortgage rates guaranteed. Apply now and get instant approval",
    "Hot singles in your area want to meet you. Click here",
    "Claim your free vacation to the Bahamas. Limited seats available",
    "Your package could not be delivered. Pay the fee at this link",
    "Increase your income now with this secret investment trick",
]
SAMPLE_HAM = [
    "Hi John, are we still meeting for lunch tomorrow at 1 pm?",
    "Please find attached the project report for this week's review",
    "Can you send me the notes from yesterday's team meeting?",
    "Reminder: the quarterly planning call is scheduled for Monday 10 am",
    "Thanks for your help with the presentation, it went really well",
    "Mom called, she wants us to come over for dinner on Sunday",
    "The invoice for last month has been approved by finance",
    "Let's catch up this weekend, I'll bring the board games",
    "Your interview is confirmed for Thursday at 3 pm in the main office",
    "Could you review my pull request when you get a chance?",
    "Happy birthday! Hope you have a fantastic day with family",
    "The meeting room has been changed to Conference Room B",
    "I have uploaded the assignment on the portal, please check",
    "Don't forget to submit your timesheet before Friday evening",
    "Sounds good, I will call you after the class ends",
]


def load_data(path=None):
    """Load a CSV (label,text) or fall back to the built-in sample set."""
    if path:
        df = pd.read_csv(path, encoding="latin-1")
        # Support common Kaggle layout: v1 = label, v2 = text
        if {"v1", "v2"}.issubset(df.columns):
            df = df.rename(columns={"v1": "label", "v2": "text"})
        df = df[["label", "text"]].dropna()
    else:
        print("[info] No dataset supplied -> using small built-in sample data.")
        df = pd.DataFrame(
            {
                "label": ["spam"] * len(SAMPLE_SPAM) + ["ham"] * len(SAMPLE_HAM),
                "text": SAMPLE_SPAM + SAMPLE_HAM,
            }
        )
    df["label"] = df["label"].str.lower().map({"spam": 1, "ham": 0})
    return df.dropna(subset=["label"]).astype({"label": int})


def clean_text(text):
    """Lowercase, replace URLs/numbers with tokens, strip punctuation."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline(model_name):
    clf = MultinomialNB() if model_name == "nb" else LogisticRegression(max_iter=1000)
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", clf),
        ]
    )


def train(data_path=None):
    df = load_data(data_path)
    df["clean"] = df["text"].apply(clean_text)
    print(f"[info] Dataset size: {len(df)} | spam ratio: {df['label'].mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    best_model, best_acc, best_name = None, -1, ""
    for name in ("nb", "lr"):
        pipe = build_pipeline(name)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n=== {'Naive Bayes' if name == 'nb' else 'Logistic Regression'} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, target_names=["ham", "spam"], zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))
        if acc > best_acc:
            best_model, best_acc, best_name = pipe, acc, name

    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[saved] Best model = {best_name.upper()} (acc {best_acc:.4f}) -> {MODEL_PATH}")


def predict(message):
    model = joblib.load(MODEL_PATH)
    cleaned = clean_text(message)
    label = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0][1]
    verdict = "SPAM" if label == 1 else "NOT SPAM (ham)"
    print(f"\nMessage : {message}\nResult  : {verdict}\nSpam probability: {prob:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spam Email Detection System")
    parser.add_argument("--data", help="Path to CSV with columns: label,text")
    parser.add_argument("--predict", help="Classify a single email/message text")
    args = parser.parse_args()

    if args.predict:
        predict(args.predict)
    else:
        train(args.data)
        # Quick demo after training
        predict("Congratulations! You won a free prize. Click http://claim.me now")
        predict("Hi team, please review the attached report before tomorrow's meeting")
