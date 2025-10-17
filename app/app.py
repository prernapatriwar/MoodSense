from flask import Flask, request, render_template
import joblib
import nltk
import re
from nltk.corpus import stopwords

# ✅ Tell Flask where your templates folder is
app = Flask(__name__, template_folder='/Users/kaustubhmestri/Projects/MoodSense/templates')

# Download stopwords (only once)
nltk.download('stopwords', quiet=True)

# ✅ Correct model loading order
model = joblib.load('/Users/kaustubhmestri/Projects/MoodSense/model/sentiment_model.pkl')
tfidf = joblib.load('/Users/kaustubhmestri/Projects/MoodSense/model/tfidf.pkl')
label_encoder = joblib.load('/Users/kaustubhmestri/Projects/MoodSense/model/label_encoder.pkl')

# Function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch.isalnum() or ch.isspace()])
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Flask will look inside the provided template_folder path

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['text']
    processed_text = preprocess_text(input_text)

    tfidf_input = tfidf.transform([processed_text])
    prediction = model.predict(tfidf_input)
    prediction_prob = model.predict_proba(tfidf_input)

    emotion = label_encoder.inverse_transform(prediction)[0]
    confidence = round(max(prediction_prob[0]) * 100, 2)

    return render_template('index.html', text=input_text, emotion=emotion, confidence=confidence)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
