Phishing Email Detection Model

A machine learning-based tool that classifies emails as Safe or Phishing using a Naive Bayes classifier combined with rule-based feature analysis.

Features:

  - ML Classification — TF-IDF vectorization + Multinomial Naive Bayes for email classification
  - Rule-Based Analysis — Keyword matching and URL pattern inspection for risk scoring
  - Risk Levels — Outputs LOW / MEDIUM / HIGH risk with explanations
  - Confidence Score — Model prediction confidence as a percentage

Requirements:

    numpy
    scikit-learn

Install dependencies:

    bashpip install numpy scikit-learn

Usage:

  Run the script directly:

      bashpython solution.py

  The model trains on built-in data, displays performance metrics, then prompts you to enter an email for analysis.

  Example

      Email Text: Urgent: Your account has been suspended. Verify now at http://secure-login.net

      ANALYSIS RESULT
      ==================================================
      Classification : Phishing
      Confidence     : 92.5%
      Risk Score     : 8/10
      Risk Level     : HIGH

      Reasons:
      ✓ Contains 1 URL(s)
      ✓ URL contains hyphens
      ✓ Uses HTTP instead of HTTPS
      ✓ Suspicious keywords: urgent, account, verify


How It Works

  1. Training (prepare_and_train)
     
      - 10 labeled emails (5 phishing, 5 safe) are vectorized with TF-IDF
      - A Multinomial Naive Bayes model is trained on a 70/30 train-test split
      - Accuracy and a confusion matrix are printed on startup


  2. Feature Analysis (analyze_email_features)

    Independently scores each email based on:

      Signal                           Points
      Contains URL(s)                  +2
      URL has hyphens                  +1
      URL is longer than 30 chars      +1
      Uses HTTP (not HTTPS)            +1
      Each phishing keyword match      +1

     Risk Level thresholds:

      Score      Level
      0 – 3      LOW
      4 – 6      MEDIUM
      7+         HIGH


  3. Classification (classify_email)

     Combines ML prediction + feature analysis to return:
      - Predicted label (Safe / Phishing)
      - Model confidence (%)
      - Risk score and level
      - Human-readable reasons



Phishing Keywords Monitored:
       verify, urgent, password, suspended, account, click, login, bank, payment, update, winner, prize, gift, alert, security, claim


Limitations:

 - Trained on only 10 examples — not suitable for production use as-is
 - Extend TRAINING_DATA with more labeled samples for better accuracy
 - Rule-based scoring is heuristic and may produce false positives
