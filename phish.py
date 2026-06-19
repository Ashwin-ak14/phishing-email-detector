import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

TRAINING_DATA = [
    ("Dear customer, your bank account has been locked. Click here to verify your identity immediately http://secure-bank-login-update.com", "Phishing"),
    ("Urgent: Your Netflix subscription has expired. Update your payment details now at http://netflix-billing-support.net", "Phishing"),
    ("Verify your account details within 24 hours to avoid suspension. Click http://paypal-security-alert.org", "Phishing"),
    ("Congratulations! You won a $1000 Amazon gift card. Claim your prize here: http://free-rewards-center.com", "Phishing"),
    ("ALERT: Unusual login activity detected on your account. Reset password via http://google-security-recovery.info", "Phishing"),
    ("Hi team, please find attached the meeting minutes from our sync earlier today. Thanks!", "Safe"),
    ("Hey, are we still on for lunch tomorrow afternoon? Let me know what time works.", "Safe"),
    ("Your monthly electricity bill is now available online. Log into your standard portal to view.", "Safe"),
    ("Hi John, thanks for sending over the project update. I will review it by Friday morning.", "Safe"),
    ("The company all-hands meeting has been rescheduled to Thursday at 10:00 AM.", "Safe")
]

PHISHING_KEYWORDS = [
    "verify", "urgent", "password", "suspended",
    "account", "click", "login", "bank",
    "payment", "update", "winner", "prize",
    "gift", "alert", "security", "claim"
]

def prepare_and_train():
    emails = [item[0] for item in TRAINING_DATA]
    labels = [item[1] for item in TRAINING_DATA]

    vectorizer = TfidfVectorizer(
        stop_words='english',
        lowercase=True
    )

    X = vectorizer.fit_transform(emails)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=["Safe", "Phishing"]
    )

    return model, vectorizer, accuracy, cm

def analyze_email_features(email_text):
    risk_score = 0
    reasons = []

    urls = re.findall(r'https?://\S+|www\.\S+', email_text)

    if urls:
        risk_score += 2
        reasons.append(f"Contains {len(urls)} URL(s)")

        for url in urls:
            if "-" in url:
                risk_score += 1
                reasons.append("URL contains hyphens")

            if len(url) > 30:
                risk_score += 1
                reasons.append("Long URL detected")

            if url.startswith("http://"):
                risk_score += 1
                reasons.append("Uses HTTP instead of HTTPS")

    detected_keywords = []

    for keyword in PHISHING_KEYWORDS:
        if keyword.lower() in email_text.lower():
            detected_keywords.append(keyword)
            risk_score += 1

    if detected_keywords:
        reasons.append(
            "Suspicious keywords: " +
            ", ".join(detected_keywords)
        )

    if risk_score <= 3:
        risk_level = "LOW"
    elif risk_score <= 6:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return risk_score, risk_level, reasons

def classify_email(email_text, model, vectorizer):
    vectorized_text = vectorizer.transform([email_text])

    prediction = model.predict(vectorized_text)[0]

    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = round(max(probabilities) * 100, 2)

    risk_score, risk_level, reasons = analyze_email_features(email_text)

    return (
        prediction,
        confidence,
        risk_score,
        risk_level,
        reasons
    )

if __name__ == "__main__":

    print("=" * 50)
    print("PHISHING EMAIL DETECTION MODEL")
    print("=" * 50)

    print("\nTraining model...")

    model, vectorizer, accuracy, cm = prepare_and_train()

    print("\nMODEL PERFORMANCE")
    print("-" * 50)
    print(f"Accuracy : {accuracy * 100:.2f}%")

    print("\nConfusion Matrix")
    print(f"[ True Safe      : {cm[0][0]} ] [ False Phishing : {cm[0][1]} ]")
    print(f"[ False Safe     : {cm[1][0]} ] [ True Phishing  : {cm[1][1]} ]")

    print("\nEnter an email to analyze")
    user_email = input("\nEmail Text: ").strip()

    if user_email:

        prediction, confidence, risk_score, risk_level, reasons = classify_email(
            user_email,
            model,
            vectorizer
        )

        print("\n" + "=" * 50)
        print("ANALYSIS RESULT")
        print("=" * 50)

        print(f"Classification : {prediction}")
        print(f"Confidence     : {confidence}%")
        print(f"Risk Score     : {risk_score}/10")
        print(f"Risk Level     : {risk_level}")

        print("\nReasons:")

        if reasons:
            for reason in reasons:
                print(f"✓ {reason}")
        else:
            print("✓ No suspicious indicators found")

        print("=" * 50)
