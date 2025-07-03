import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# loading the dataset in dataframe
df = pd.read_csv('movies.csv')

#loading information

print(df.info())

#filter the required for recommendation

required_coloum = ['genres', 'keywords','overview', 'title']

#updated dataframe
df = df[required_coloum]
print(df.shape)

#checking the missing value

print(df.isnull().sum())

#droping all the null values not the best method but work efficient 
df = df.dropna().reset_index(drop = True)

df['combined'] = df['genres']+'' + df['keywords']+''+df['overview']


#used .copy for resolving issue of copywriting
'''This data is not guaranteed to be a standalone copy —
it's a view of df, so when you try to add a new column ('cleaned_text')
to it, Pandas warns that this might not behave as expected.
'''
data = df[['title', 'combined']].copy()

#Wordcloud for movie content
combined_text = "".join(df['combined'])
wordCloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)


#wordcloud to visualize the most common words in the movie content

plt.figure(figsize=(10,5))
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title("most common words in movie content")
plt.show()

#required NLTK data 
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

#defining function for preprocessing data

def preprocess_data(text):
    #remove special character and numbers
    text = re.sub(r"[^a-zA-Z\s]","",text)

    #convert to lowercase
    text = text.lower()

    #Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

#apply preprocessing to the movie content

data['cleaned_text']= df['combined'].apply(preprocess_data)

print(data.head())

#vectorize with tf-idf

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_text'])

#compute cosine similarity 
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# recommendation function 
 