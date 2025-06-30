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

#leading information 

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

data = df[['title', 'combined']]

#Wordcloud for movie content
combined_text = "".join(df['combined'])
wordCloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)


#wordcloud to visualize the most common words in the movie content

plt.figure(figsize=(10,5))
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.title("most common words in movie content")
plt.show()