import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split  # Added import
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming the Word2VecTransformer is correctly implemented in your 'linearsvcw2v' module
from linearsvcw2v import Word2VecTransformer  # If necessary, use this class

# Load the labeled dataset
df = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\vaccination_tweets_labeled.csv')

# Handle missing values in the 'text' column
df['text'] = df['text'].fillna('')

# Assuming the dataset has a 'text' column for the tweet text and a 'sentiment' column for sentiment labels
X = df['text']  # The text data
y = df['sentiment']  # The sentiment labels

# Split the data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved pipeline (which includes both the vectorizer and LinearSVC model)
model_path = r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\svc_model_80_20.pkl'
with open(model_path, 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)  # Using the test set for evaluation

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on 30% test set: {accuracy:.4f}")


# Generate the classification report (precision, recall, f1-score)
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# You can also extract individual components if needed
report_dict = classification_report(y_test, y_pred, output_dict=True)
print(confusion_matrix(y_test, y_pred))

# For example, extract precision, recall, and f1 score for each class:
# precision = {label: report_dict[label]['precision'] for label in report_dict if label not in ['accuracy', 'macro avg', 'weighted avg']}
# recall = {label: report_dict[label]['recall'] for label in report_dict if label not in ['accuracy', 'macro avg', 'weighted avg']}
# f1_score = {label: report_dict[label]['f1-score'] for label in report_dict if label not in ['accuracy', 'macro avg', 'weighted avg']}

print("Precision per class:", precision)
print("Recall per class:", recall)
print("F1-score per class:", f1_score)
