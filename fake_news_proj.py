#tester tester 
import re
from cleantext import clean
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


fake_news = pd.read_csv('https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv')

ps = PorterStemmer()

#print(new_data)
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Replace URLs
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = re.sub(r'<URL>', '<URL>', text)
    # Replace emails
    email_pattern = re.compile(r'\S+@\S+')
    text = re.sub(r'<EMAIL>', '<EMAIL>', text)

    # Replace dates (YYYY-MM-DD and DD/MM/YYYY formats)
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})')
    text = date_pattern.sub('<DATE>', text)

    # Replace numbers
    num_pattern = re.compile(r'\b\d+\b')
    text = re.sub(r'<NUMBER>', '<NUMBER>', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)

    return text

fake_news['content_clean_maual'] = fake_news['content'].apply(clean_text)
# Load the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


fake_news['content_clean_manual'] = fake_news['content'].apply(clean_text)

# Tokenization
fake_news['content_tokens'] = fake_news['content_clean_manual'].apply(word_tokenize)

# Stemming
fake_news['content_stemming'] = fake_news['content_tokens'].apply(lambda x: [ps.stem(word) for word in x])
# Stop word removal
stop_words = set(stopwords.words('english'))
fake_news['content_clean'] = fake_news['content_stemming'].apply(lambda x: [word for word in x if word not in stop_words])

# Calculating the number of unique words in the data after preprocessing
#cleaned_text = fake_news['content_clean'].explode().unique()
#num_unique_words_after_preprocessing = len(cleaned_text)

# Calculating how frequently each of these words is used in the dataset
#word_counts = fake_news['content_clean'].explode().value_counts()

# Sort this list, so that the most frequent word appears first
#word_counts = word_counts.sort_values(ascending=False)

# using matplotlib to plot the data
#plt.figure(figsize=(15, 10))
#plt.barh(word_counts.index[:50],word_counts.values[:50])
#plt.xlabel('Frequency')
#plt.ylabel('Word')
#plt.title('50 Most Frequent Words in the Dataset')
#plt.show()

