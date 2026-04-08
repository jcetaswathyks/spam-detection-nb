"""
Email Spam Detector
====================
Uses TF-IDF vectorization + Multinomial Naive Bayes classifier.
Includes training on a built-in sample dataset, evaluation metrics,
and a live prediction function for new emails.

Dependencies:
    pip install scikit-learn
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────
# 1. Sample Dataset
# ─────────────────────────────────────────────

SAMPLE_EMAILS = [
    # SPAM
    ("Congratulations! You've won a $1,000 gift card. Click here to claim now!", "spam"),
    ("FREE OFFER: Get a free iPhone. Limited time only. Act fast!", "spam"),
    ("You have been selected for a special prize. Send your details to claim.", "spam"),
    ("Buy cheap meds online. No prescription needed. 90% off!", "spam"),
    ("Make money fast working from home. Earn $5000 per week guaranteed!", "spam"),
    ("URGENT: Your bank account has been compromised. Verify now!", "spam"),
    ("Hot singles in your area want to meet you. Click here!", "spam"),
    ("Lose 20 pounds in 2 weeks with this miracle pill. Order now!", "spam"),
    ("You are the lucky winner of our monthly lottery draw!", "spam"),
    ("Get cheap loans approved instantly. No credit check required.", "spam"),
    ("Increase your income by 500%. Join our exclusive network today!", "spam"),
    ("WARNING: Your computer is infected. Download our free antivirus!", "spam"),
    ("Nigerian prince needs your help to transfer $10 million. Share details.", "spam"),
    ("Click here for free adult content. No registration required.", "spam"),
    ("Earn passive income with crypto. 100% profit guaranteed.", "spam"),

    # HAM (legitimate)
    ("Hi John, can we schedule a meeting for Thursday afternoon?", "ham"),
    ("Please find attached the project report for your review.", "ham"),
    ("Reminder: Your dentist appointment is tomorrow at 10am.", "ham"),
    ("The team lunch is scheduled for Friday at noon in the cafeteria.", "ham"),
    ("Thanks for your help yesterday. I really appreciated it.", "ham"),
    ("Can you send me the latest version of the presentation?", "ham"),
    ("Your Amazon order has shipped. Expected delivery: 2 days.", "ham"),
    ("Just checking in — how are you doing with the deadline?", "ham"),
    ("Happy birthday! Hope you have a wonderful day.", "ham"),
    ("The server maintenance is scheduled for Sunday night.", "ham"),
    ("Please review the attached contract and let me know your thoughts.", "ham"),
    ("Mom asked if you're coming home for the holidays.", "ham"),
    ("Your flight booking is confirmed. Check-in opens 24 hours before.", "ham"),
    ("The quarterly results meeting has been moved to 3pm.", "ham"),
    ("I wanted to follow up on our conversation from last week.", "ham"),
    ("Great work on the presentation today. The client loved it.", "ham"),
    ("Could you help me debug this Python function? I'm getting a TypeError.", "ham"),
    ("Meeting notes from today's standup are attached.", "ham"),
    ("Reminder: Submit your timesheet by end of day Friday.", "ham"),
    ("Library book due in 3 days. Renew online to avoid late fees.", "ham"),
]


# ─────────────────────────────────────────────
# 2. Text Preprocessing
# ─────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)   # replace URLs
    text = re.sub(r"\d+", " num ", text)               # replace numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# 3. Train the Model
# ─────────────────────────────────────────────

def train_model(dataset):
    """
    Build and train a TF-IDF + Naive Bayes pipeline.

    Args:
        dataset: list of (text, label) tuples where label is 'spam' or 'ham'

    Returns:
        pipeline: trained sklearn Pipeline
        X_test, y_test: held-out test data for evaluation
    """
    texts, labels = zip(*dataset)
    texts = [preprocess(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams
            max_df=0.95,
            min_df=1,
            sublinear_tf=True
        )),
        ("clf", MultinomialNB(alpha=0.5))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


# ─────────────────────────────────────────────
# 4. Evaluate the Model
# ─────────────────────────────────────────────

def evaluate(pipeline, X_test, y_test):
    """Print evaluation metrics on the test set."""
    y_pred = pipeline.predict(X_test)

    print("=" * 50)
    print("         MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.2%}")
    print(f"  Precision : {precision_score(y_test, y_pred, pos_label='spam'):.2%}")
    print(f"  Recall    : {recall_score(y_test, y_pred, pos_label='spam'):.2%}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred, pos_label='spam'):.2%}")
    print()

    cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"            {'SPAM':>8}  {'HAM':>6}")
    print(f"  SPAM  →   {cm[0][0]:>6}    {cm[0][1]:>4}")
    print(f"  HAM   →   {cm[1][0]:>6}    {cm[1][1]:>4}")
    print()
    print("  Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    print("=" * 50)


# ─────────────────────────────────────────────
# 5. Predict New Emails
# ─────────────────────────────────────────────

def predict(pipeline, email_text: str) -> dict:
    """
    Classify a single email as spam or ham.

    Args:
        pipeline: trained sklearn Pipeline
        email_text: raw email string

    Returns:
        dict with 'label', 'confidence_spam', 'confidence_ham'
    """
    processed = preprocess(email_text)
    label = pipeline.predict([processed])[0]
    proba = pipeline.predict_proba([processed])[0]
    classes = pipeline.classes_

    confidence = {cls: round(prob * 100, 2) for cls, prob in zip(classes, proba)}

    return {
        "label": label.upper(),
        "confidence_spam": confidence.get("spam", 0),
        "confidence_ham": confidence.get("ham", 0),
    }


def display_prediction(email_text: str, result: dict):
    tag = "🚨 SPAM" if result["label"] == "SPAM" else "✅ HAM (Legitimate)"
    print(f"\nEmail   : \"{email_text[:70]}{'...' if len(email_text) > 70 else ''}\"")
    print(f"Result  : {tag}")
    print(f"Spam    : {result['confidence_spam']}%  |  Ham: {result['confidence_ham']}%")


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\nTraining spam detector...")
    model, X_test, y_test = train_model(SAMPLE_EMAILS)
    print("Training complete.\n")

    evaluate(model, X_test, y_test)

    # ── Test with new emails ──
    test_emails = [
        "You've won a FREE vacation! Click now to claim your prize.",
        "Hey, are you free for a quick call tomorrow morning?",
        "URGENT: Verify your PayPal account immediately to avoid suspension.",
        "The sprint retrospective is moved to 4pm on Friday.",
        "Buy 2 get 3 free! Massive sale on all products. Limited time!",
        "Can you review my pull request when you get a chance?",
    ]

    print("\n─── Live Predictions ───")
    for email in test_emails:
        result = predict(model, email)
        display_prediction(email, result)

    # ── Interactive Mode ──
    print("\n\n─── Interactive Mode (type 'quit' to exit) ───")
    while True:
        user_input = input("\nEnter email text: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting. Goodbye!")
            break
        if not user_input:
            continue
        result = predict(model, user_input)
        display_prediction(user_input, result)