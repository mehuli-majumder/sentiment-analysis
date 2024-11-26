import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
import pickle

# Load the labeled dataset
df = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\vaccination_tweets_labeled.csv')

df = df.dropna(subset=['text'])

# Assuming the dataset has a 'text' column for the tweet text and a 'label' column for sentiment labels
X = df['text']  # The text data
y = df['sentiment']  # The sentiment labels

# Preprocess the text (Tokenize and clean the text)
def preprocess_text(text):
    # Tokenize by splitting the text into words and converting to lowercase
    return text.lower().split()

# Create a custom transformer to convert text to Word2Vec embeddings
class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Preprocess the text and get the Word2Vec embeddings for each tweet
        return np.array([self.get_vector(text) for text in X])

    def get_vector(self, text):
        # Tokenize the text
        words = preprocess_text(text)
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

# Load a pre-trained Word2Vec model (ensure the correct path to your model)
word2vec_model = Word2Vec.load(r"C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\word2vec.model")

# Create the pipeline with Word2Vec embeddings and LinearSVC
pipeline = make_pipeline(
    Word2VecTransformer(model=word2vec_model),
    LinearSVC(class_weight="balanced", random_state=42)
)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# Print the results of the cross-validation
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {np.mean(cv_scores)}")
print(f"Standard deviation of cross-validation accuracy: {np.std(cv_scores)}")

# Save the model after cross-validation
pipeline.fit(X, y)  # Fit the model on the entire dataset after cross-validation
with open('svc_model_with_word2vec_cv.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

# 70-30 Split
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train_70, y_train_70)
accuracy_70_30 = pipeline.score(X_test_30, y_test_30)
print(f"70-30 split accuracy: {accuracy_70_30}")

# Save the 70-30 split model
with open('svc_model_with_word2vec_70_30.pkl', 'wb') as model_file_70_30:
    pickle.dump(pipeline, model_file_70_30)

# 80-20 Split
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train_80, y_train_80)
accuracy_80_20 = pipeline.score(X_test_20, y_test_20)
print(f"80-20 split accuracy: {accuracy_80_20}")

# Save the 80-20 split model
with open('svc_model_with_word2vec_80_20.pkl', 'wb') as model_file_80_20:
    pickle.dump(pipeline, model_file_80_20)
