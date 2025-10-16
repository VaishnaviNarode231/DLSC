import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import pickle

# Parameters
vocab_size = 10000
max_len = 200
embedding_dim = 128

# 1. Load IMDB dataset
print("Loading IMDB dataset...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# 2. Pad sequences
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# 3. Build LSTM model
print("Building model...")
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# 4. Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train with validation split + history tracking
print("Training model...")
history = model.fit(
    x_train, y_train,
    epochs=5,                   # increased epochs for better learning
    batch_size=64,
    validation_split=0.2,
    verbose=1                   # shows live progress per epoch
)

# 6. Evaluate
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"\n✅ Test Accuracy: {accuracy*100:.2f}%")

# 7. Save the model
model.save("sentiment_lstm_model.h5")
print("Model saved as sentiment_lstm_model.h5")

# 8. Save word index for preprocessing input text
word_index = imdb.get_word_index()
with open("word_index.pkl", "wb") as f:
    pickle.dump(word_index, f)
print("Word index saved as word_index.pkl")

# 9. Plot accuracy & loss
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
