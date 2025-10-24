import joblib
import json
import random
from preprocessing import custom_preprocessor
import numpy as np

# ----------------- Load trained model and encoder -----------------
chat_model = joblib.load("chatbot_model.joblib")
le = joblib.load("label_encoder.joblib")

# ----------------- Load intents -----------------
with open("intents.json", 'r', encoding='utf-8') as f:
    intents_data = json.load(f)

# Dictionary: tag -> responses
tag_responses = {intent["tag"]: intent["responses"] for intent in intents_data["intents"]}

# ----------------- Chatbot response function -----------------
def get_response(user_input, confidence_threshold=0.03):
    try:
        # Preprocess user input
        processed_input = custom_preprocessor(user_input)

        # Predict encoded tag
        pred_encoded = chat_model.predict([processed_input])[0]

        # Prediction probabilities
        probs = chat_model.predict_proba([processed_input])[0]
        confidence = np.max(probs)

        # Decode tag
        pred_tag = le.inverse_transform([pred_encoded])[0]

        # Choose response
        if confidence >= confidence_threshold:
            return random.choice(tag_responses.get(pred_tag, ["I hear you."]))
        else:
            return random.choice(tag_responses.get('no-response', ["Sorry, I didn't understand."]))

    except Exception as e:
        print(f"RUNTIME ERROR: {e}")
        return "Technical issue. Please try again."
