import joblib
import numpy as np
from clean_text import clean_text

model = joblib.load('sentiment_model.joblib')

def load_glove_embeddings(glove_path, embedding_dim):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != embedding_dim + 1:
                continue
            word = parts[0]
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector
            except ValueError:
                continue
    return embeddings

def text_to_avg_vector(text, embeddings, embedding_dim):
    words = text.split()
    valid_vectors = []
    for word in words:
        vec = embeddings.get(word)
        if vec is not None and len(vec) == embedding_dim:
            valid_vectors.append(vec)
    if not valid_vectors:
        return np.zeros((1, embedding_dim))
    avg_vector = np.mean(valid_vectors, axis=0)
    return avg_vector.reshape(1, -1)

glove_path = 'glove.txt'
embedding_dim = 100
embeddings_index = load_glove_embeddings(glove_path, embedding_dim)

text = '1'
while text:
    text = input("Enter a message to test (empty to quit):\n")
    if not text:
        break

    cleaned = clean_text(text)
    vectorized = text_to_avg_vector(cleaned, embeddings_index, embedding_dim)

    prediction = model.predict(vectorized)
    if prediction == 1:
        print("Positive")
    else:
        print("Negative")

