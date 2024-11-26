import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
import numpy as np
import pickle

# Load the labeled dataset
df = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\vaccination_tweets_labeled.csv')

df = df.dropna(subset=['text'])

# Assuming the dataset has a 'tweet' column for the tweet text and a 'label' column for sentiment labels
X = df['text']  # The text data
y = df['sentiment']  # The sentiment labels

# Convert text data into TF-IDF features and train the LinearSVC using a pipeline
pipeline = make_pipeline(
    TfidfVectorizer(max_features=5000),  # You can adjust the number of features
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
with open('svc_model_cv.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

# # 70-30 Split
# X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, test_size=0.3, random_state=42)
# pipeline.fit(X_train_70, y_train_70)
# accuracy_70_30 = pipeline.score(X_test_30, y_test_30)
# print(f"70-30 split accuracy: {accuracy_70_30}")

# # Save the 70-30 split model
# with open('svc_model_70_30.pkl', 'wb') as model_file_70_30:
#     pickle.dump(pipeline, model_file_70_30)

# 80-20 Split
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train_80, y_train_80)
accuracy_80_20 = pipeline.score(X_test_20, y_test_20)
print(f"80-20 split accuracy: {accuracy_80_20}")

# Save the 80-20 split model
with open('svc_model_80_20.pkl', 'wb') as model_file_80_20:
    pickle.dump(pipeline, model_file_80_20)
