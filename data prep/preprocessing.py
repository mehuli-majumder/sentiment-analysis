import pandas as pd
import re
import ast
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def add_raw_text_features(df):
    """Extract punctuation and capitalization features."""
    df['exclamation_count'] = df['text'].apply(lambda x: x.count('!'))
    df['question_count'] = df['text'].apply(lambda x: x.count('?'))
    df['has_all_caps'] = df['text'].apply(lambda x: 1 if any(word.isupper() for word in x.split()) else 0)
    return df

def clean_and_lemmatize_dataset(df):
    """Clean and lemmatize the dataset."""
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].apply(lambda x: re.sub('@[^\s]+', '', x))  # Remove mentions
    df['text'] = df['text'].apply(lambda x: re.sub(r'\B#\S+', '', x))  # Remove hashtags
    df['text'] = df['text'].apply(lambda x: re.sub(r"http\S+", "", x))  # Remove URLs
    df['text'] = df['text'].apply(lambda x: ' '.join(re.findall(r'\w+', x)))  # Keep only words
    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) 
                                                      for word in x.split() if word not in stop_words]))
    return df

def add_domain_specific_features(df):
    """Extract domain-specific hashtag features."""
    vaccine_keywords = ['GetVaccinated', 'PfizerBioNTech', 'VaccinesWork', 'covid19', 
                        'CovidVaccine', 'vaccination', 'COVID19Vaccine', 'Vaccine', 'coronavirus']
    df['hashtags'] = df['hashtags'].fillna('[]')  # Replace NaN with empty lists
    df['has_vaccine_keyword'] = df['hashtags'].apply(
        lambda x: 1 if any(keyword.lower() in [tag.lower() for tag in ast.literal_eval(x)] 
                           for keyword in vaccine_keywords) else 0)
    return df

def add_vader_sentiment(df):
    """Add VADER sentiment scores."""
    analyzer = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    return df

# Load the dataset
vaccination_tweets = pd.read_csv(r'C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\linearsvc\vaccination_tweets.csv')

# Apply preprocessing
vaccination_tweets = add_raw_text_features(vaccination_tweets)
vaccination_tweets = clean_and_lemmatize_dataset(vaccination_tweets)
vaccination_tweets = add_domain_specific_features(vaccination_tweets)
vaccination_tweets = add_vader_sentiment(vaccination_tweets)

# Save the preprocessed dataset
vaccination_tweets.to_csv('vaccination_tweets_preprocessed.csv', index=False)

print("Preprocessing completed. Preprocessed data saved to 'vaccination_tweets_preprocessed.csv'.")
