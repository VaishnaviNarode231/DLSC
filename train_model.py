import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load model and word index
model = tf.keras.models.load_model("sentiment_lstm_model.h5")
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

vocab_size = 10000
max_len = 200

def preprocess_text_simple(text):
    text = text.lower().replace("<br />", " ")
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = []
    for w in text.split():
        idx = word_index.get(w)
        if idx is None:
            tokens.append(2)
        else:
            idx = idx + 3
            if idx >= vocab_size:
                tokens.append(2)
            else:
                tokens.append(idx)
    return pad_sequences([tokens], maxlen=max_len)

samples = [
    "I absolutely loved this movie, the acting was brilliant and the story was touching.",
    "Terrible product, it broke after one day of use.",
    "The service was slow and the staff was rude.",
    "It was an okay film, but too long for my taste.",
    "One of the best experiences I've ever had.",
    "I regret buying this, waste of money."
]

for s in samples:
    seq = preprocess_text_simple(s)
    prob = float(model.predict(seq)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    print(f"{label} ({prob:.3f}) — \"{s}\"")
