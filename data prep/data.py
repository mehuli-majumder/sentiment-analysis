import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv(r"C:\Users\MEHULI MAJUMDER\OneDrive\Desktop\code stuff\bos sentiment analysis\starting afresh\data prep\vaccination_tweets_labeled.csv")

# Using a for loop to filter text for 'anger' sentiment
texts = []
count=0
for _, row in df.iterrows():
    if row['sentiment'] == 'NEUTRAL':
        texts.append(row['text'])
for i in texts:
    count+=1
# Display the collected texts
print(texts)
print(count)
#print(df.columns)