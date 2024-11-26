import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\vaccination_tweets_labeled.csv')

df = df.dropna(subset=['text'])

# Assuming the dataset has a 'text' column for the tweet text and a 'label' column for sentiment labels
X = df['text']  # The text data
y = df['sentiment']  # The sentiment labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the pre-trained model
with open('rf_model_80_20.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Split the data (70-30 split for evaluation)
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
