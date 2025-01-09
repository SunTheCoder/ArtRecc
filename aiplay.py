import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Step 1: Parse the JSONL data
data = []
with open('data_converted.jsonl', 'r') as file:  # Replace with your actual file name
    for line in file:
        entry = json.loads(line)
        # Join the mood array into a string
        mood_text = ", ".join(entry["contents"][0].get("mood", []))
        user_text = entry["contents"][0]["parts"][0]["text"] + " mood: " + mood_text
        artID = entry["contents"][1]["parts"][1]["artID"][0]  # Extract the first ArtID
        reply = entry["contents"][1]["parts"][0]["text"]
        data.append({"text": user_text, "artID": artID})

# Step 2: Prepare data for training
texts = [entry["text"] for entry in data]
artIDs = [entry["artID"] for entry in data]

# Tokenize the texts
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")  # Limit vocabulary size for small datasets
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=10)

# Map ArtIDs to integers
artID_to_index = {artID: idx for idx, artID in enumerate(set(artIDs))}
index_to_artID = {idx: artID for artID, idx in artID_to_index.items()}
y = np.array([artID_to_index[artID] for artID in artIDs])

# Convert output to one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=len(artID_to_index))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Step 3: Load GloVe embeddings
embedding_dim = 200
embedding_index = {}
with open("glove.6B.200d.txt", "r", encoding="utf-8") as f:  # Replace with the path to your GloVe file
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefficients

# Create embedding matrix
word_index = tokenizer.word_index
num_words = min(1000, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < num_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Step 4: Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_words, 
                               output_dim=embedding_dim, 
                               weights=[embedding_matrix], 
                               input_length=10, 
                               trainable=False),  # Use pre-trained embeddings
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),  # Bi-directional LSTM
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Dropout for regularization
    tf.keras.layers.Dense(len(artID_to_index), activation='softmax')
])

# Compile the model with a lower learning rate for stability
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Fine-tuned learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate scheduler
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1
)

# Step 5: Train the model
history = model.fit(
    X_train, y_train,
    epochs=15,  # Increased epochs for better learning
    validation_data=(X_test, y_test),
    batch_size=16,  # Smaller batch size for small datasets
    verbose=2,
    callbacks=[lr_scheduler]  # Add the learning rate scheduler here
)

# Step 6: Test the model
test_text = ["The weather is nice! mood: happy, excited"]  # Example test input
test_seq = tokenizer.texts_to_sequences(test_text)
test_padded = pad_sequences(test_seq, padding='post', maxlen=20)
predicted_class = np.argmax(model.predict(test_padded), axis=-1)

# Get the predicted ArtID
predicted_artID = index_to_artID[predicted_class[0]]
print(f"Recommended ArtID: {predicted_artID}")
