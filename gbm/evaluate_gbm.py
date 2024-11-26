import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\gbm\vaccination_tweets_labeled.csv')

# Drop rows with missing text
df = df.dropna(subset=['text'])

# Assuming the dataset has 'text' for tweets and 'sentiment' for labels
X = df['text']
y = df['sentiment']

# Split the data into training and testing sets (80-20 split for training and testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load the pre-trained Gradient Boosting model (choose the appropriate model)
with open('gbm_model_70_30.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Make predictions on the test dataset
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
