import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import random

def load_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
    return texts

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
        return np.zeros(embedding_dim)
    return np.mean(valid_vectors, axis=0)

# Load embeddings
glove_path = 'glove.txt'
embedding_dim = 100
embeddings_index = load_glove_embeddings(glove_path, embedding_dim)
print(f"Loaded {len(embeddings_index)} word vectors.")

# Load data
positive_train_texts = load_texts_from_folder("cleaned_data/train/pos")
negative_train_texts = load_texts_from_folder("cleaned_data/train/neg")
positive_test_texts = load_texts_from_folder("cleaned_data/test/pos")
negative_test_texts = load_texts_from_folder("cleaned_data/test/neg")

train_texts = positive_train_texts + negative_train_texts
test_texts = positive_test_texts + negative_test_texts

# Vectorize text
X_train = np.array([text_to_avg_vector(text, embeddings_index, embedding_dim) for text in train_texts])
X_test = np.array([text_to_avg_vector(text, embeddings_index, embedding_dim) for text in test_texts])

# Labels
n_pos_train = len(positive_train_texts)
n_train = len(train_texts)
y_train = np.zeros(n_train, dtype=np.int8)
y_train[:n_pos_train] = 1

n_pos_test = len(positive_test_texts)
n_test = len(test_texts)
y_test = np.zeros(n_test, dtype=np.int8)
y_test[:n_pos_test] = 1

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1-score:      {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()
plt.show()

print(f"AUC Score: {auc_score:.4f}")

# Example predictions
print("\nExample Predictions:")
sample_indices = random.sample(range(len(X_test)), 5)
for idx in sample_indices:
    true_label = "Positive" if y_test[idx] == 1 else "Negative"
    pred_label = "Positive" if y_pred[idx] == 1 else "Negative"
    confidence = y_pred_proba[idx]
    print(f"\nText: {test_texts[idx][:200]}...")
    print(f"True Label: {true_label}")
    print(f"Predicted: {pred_label} (Confidence: {confidence:.2f})")

# Save model
joblib.dump(model, 'sentiment_model.joblib')
print("Model saved as sentiment_model.joblib")
