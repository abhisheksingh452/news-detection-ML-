from flask import Flask, render_template, request
import pickle
import re

# Initialize app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.form["news"]
    
    # Clean text
    cleaned_data = clean_text(data)
    
    # Convert to vector
    vector = vectorizer.transform([cleaned_data])
    
    # Predict
    prediction = model.predict(vector)[0]

    # Output formatting
    if prediction == 0:
        output = "❌ Fake News"
    else:
        output = "✅ Real News"

    return render_template("index.html", prediction=output)

# Run app
if __name__ == "__main__":
    app.run(debug=True)