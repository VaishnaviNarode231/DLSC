from flask import Flask, render_template, request, url_for
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

# Disable TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ✅ Use non-GUI backend for Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load trained model and word index
model = tf.keras.models.load_model("sentiment_lstm_model.h5")
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

# Params
vocab_size = 10000
max_len = 200

# Ensure static directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Preprocessing function (fixed +3 offset)
def preprocess_text(text, word_index, max_len=max_len, vocab_size=vocab_size):
    text = text.lower().replace("<br />", " ")
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = []
    for w in text.split():
        if not w:
            continue
        idx = word_index.get(w)
        if idx is None:
            tokens.append(2)  # <UNK>
        else:
            idx = idx + 3     # Important fix!
            if idx >= vocab_size:
                tokens.append(2)
            else:
                tokens.append(idx)
    return pad_sequences([tokens], maxlen=max_len)

# Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    chart_path = None

    if request.method == "POST":
        text = request.form["text"]
        seq = preprocess_text(text, word_index)
        prob = float(model.predict(seq)[0][0])
        confidence = round(prob, 3)
        result = "Positive 😀" if prob >= 0.5 else "Negative 😡"

        # ✅ Generate chart path dynamically inside /static
        chart_path = os.path.join(static_dir, "confidence.png")

        # Plot confidence bar chart
        plt.figure(figsize=(3, 2))
        plt.bar(["Negative", "Positive"], [1 - confidence, confidence], color=['red', 'green'])
        plt.title("Sentiment Confidence")
        plt.savefig(chart_path)
        plt.close()

    return render_template("index.html", result=result, confidence=confidence, chart=url_for('static', filename='confidence.png'))

if __name__ == "__main__":
    app.run(debug=True)
