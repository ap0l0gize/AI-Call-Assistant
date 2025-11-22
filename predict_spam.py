import json
import string
import pickle
from nltk.corpus import stopwords

def predict_spam():
    # load json input
    with open("user_data.json", "r") as f:
        user_data = json.load(f)

    reason_for_call = user_data["reason"]

    # preprocess text like in training
    stopwords_set = set(stopwords.words('english'))
    text = reason_for_call.lower()
    text = text.translate(str.maketrans('', '', string.punctuation)).split()
    text = [word for word in text if word not in stopwords_set]
    text = ' '.join(text)

    # load model and vectorizer
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # transform text into numeric features
    X_new = vectorizer.transform([text])

    # predict
    prediction = model.predict(X_new)
    print("Prediction:", "spam" if prediction[0] == 1 else "ham")
