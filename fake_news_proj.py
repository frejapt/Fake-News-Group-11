#tester tester 
import re
from cleantext import clean
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

fake_news = pd.read_csv('https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv')

#clean_data=[]
#for i in range(len(fake_news)):
new_data= clean(fake_news, 
            lower=True,
            fix_unicode=True,               
            to_ascii=True,
            no_line_breaks=True,
            no_urls=True,                  
            no_emails=True,               
            no_phone_numbers=True,         
            no_numbers=True,               
            no_digits=True,
            normalize_whitespace=True,  
            no_punct=True,
            replace_with_punct="",                      
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUM>",
            replace_with_digit="0",
            lang="en"
            )
   #clean_data.append(new_data)
   
print(new_data)

# Load the necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')


fake_news['content_clean_manual'] = fake_news['content'].apply(clean_text)

# Tokenization
fake_news['content_tokens'] = fake_news['content_clean_manual'].apply(word_tokenize)

# Stop word removal
stop_words = set(stopwords.words('english'))
fake_news['content_clean'] = fake_news['content_tokens'].apply(lambda x: [word for word in x if word not in stop_words])
