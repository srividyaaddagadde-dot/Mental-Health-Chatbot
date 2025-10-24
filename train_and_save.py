# train_chatbot_json.py

import json
import joblib
from preprocessing import custom_preprocessor  # your preprocessing function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ----------------- Load JSON -----------------
with open("intents.json", 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

patterns, tags = [], []

# Collect patterns and tags from intents
for intent in intents_data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(custom_preprocessor(pattern))  # Preprocess here
        tags.append(intent["tag"])

# ----------------- Encode labels -----------------
le = LabelEncoder()
y = le.fit_transform(tags)

# ----------------- Train/test split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    patterns, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------- Build and train model -----------------
# TF-IDF uses default tokenizer since text is already preprocessed
chat_model = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=None)),  
    ('clf', MultinomialNB())
])
chat_model.fit(X_train, y_train)

# ----------------- Evaluate -----------------
y_pred = chat_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(
    y_test, y_pred,
    labels=range(len(le.classes_)),
    target_names=le.classes_,
    zero_division=0
))

# ----------------- Save model and label encoder -----------------
joblib.dump(chat_model, "chatbot_model.joblib")
joblib.dump(le, "label_encoder.joblib")
print("Model and label encoder saved successfully!")
