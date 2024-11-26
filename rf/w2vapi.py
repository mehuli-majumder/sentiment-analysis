from gensim.models import Word2Vec
import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\vaccination_tweets_labeled.csv')

# Drop rows with missing text data
df = df.dropna(subset=['text'])

# Ensure the 'text' column contains strings (in case there are any non-string values)
df['text'] = df['text'].astype(str)

# Preprocess the text data (tokenize)
tokenized_text = [text.split() for text in df['text']]

# Train a Word2Vec model
word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

# Save the model for later use
word2vec_model.save("word2vec.model")

