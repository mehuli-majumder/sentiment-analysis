import pandas as pd
from transformers import pipeline
import numpy as np

def label_dataset_with_sentiment_analysis(input_file, output_file):
    """Label the dataset with sentiment analysis using DistilBERT."""
    # Load the preprocessed dataset
    df = pd.read_csv(input_file)
    
    # Initialize Hugging Face sentiment analysis pipeline
    sentiment_classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    def classify_sentiment(text):
        """Classify sentiment for a given text."""
        # Skip NaN values
        if pd.isna(text):
            return "NEUTRAL"  # You can also use 'return np.nan' or similar if you prefer
        
        result = sentiment_classifier(text)
        label = result[0]['label']
        score = result[0]['score']
        
        # Neutral threshold (e.g., score < 0.6)
        if score < 0.6:
            return "NEUTRAL"
        
        return label

    # Apply sentiment classification and skip rows with NaN values
    df['sentiment'] = df['text'].apply(classify_sentiment)
    
    # Remove rows where sentiment is NaN (or None)
    df.dropna(subset=['sentiment'], inplace=True)
    
    # Save the labeled dataset
    df.to_csv(output_file, index=False)
    print(f"Labeled dataset saved to {output_file}")

# Specify file paths
input_file = r"C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\data prep\vaccination_tweets_preprocessed.csv"
output_file = "vaccination_tweets_labeled.csv"

# Run the labeling process
if __name__ == "__main__":
    label_dataset_with_sentiment_analysis(input_file, output_file)
