import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Define Word2VecTransformer class (same as in training script)
from sklearn.base import BaseEstimator, TransformerMixin

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Preprocess the text and get the Word2Vec embeddings for each tweet
        return np.array([self.get_vector(text) for text in X])

    def get_vector(self, text):
        words = text.lower().split()
        word_vecs = []
        
        for word in words:
            try:
                word_vec = self.model.wv[word]
                word_vecs.append(word_vec)
            except KeyError:
                # If the word is not in the model's vocabulary, use a zero vector
                word_vecs.append(np.zeros(self.model.vector_size))
        
        if word_vecs:
            # Return the average of all the word vectors in the tweet
            return np.mean(word_vecs, axis=0)
        else:
            # If no words found in the model, return a zero vector
            return np.zeros(self.model.vector_size)

# Load the evaluation dataset
df = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\gbm\vaccination_tweets_labeled.csv')
df = df.dropna(subset=['text'])

# Assuming the dataset has a 'text' column for the tweet text and a 'label' column for sentiment labels
X = df['text']  # The text data
y = df['sentiment']  # The sentiment labels

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load the saved GBM model (change file path as needed to evaluate different models)
model_path = r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\gbm\gbm_model_with_word2vec_70_30.pkl'
with open(model_path, 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on 70-30 split: {accuracy:.2f}")

# Generate classification report
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)
