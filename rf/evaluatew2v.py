import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec

# Define the Word2VecTransformer class
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self.get_vector(text) for text in X])

    def get_vector(self, text):
        words = text.lower().split()
        word_vecs = []
        for word in words:
            try:
                word_vec = self.model.wv[word]
                word_vecs.append(word_vec)
            except KeyError:
                word_vecs.append(np.zeros(self.model.vector_size))
        return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(self.model.vector_size)

# Load the dataset
df = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\vaccination_tweets_labeled.csv')

# Drop rows with missing text data
df = df.dropna(subset=['text'])

# Split the dataset into features (X) and labels (y)
X = df['text']
y = df['sentiment']

# Load the saved model (ensure the correct path to your model file)
model_file_path = r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\rf\rf_model_with_word2vec_80_20.pkl'
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Predict on the dataset
y_pred = model.predict(X)

# Calculate and print the accuracy score
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")

# Generate and print the classification report
report = classification_report(y, y_pred)
print("Classification Report:")
print(report)

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
