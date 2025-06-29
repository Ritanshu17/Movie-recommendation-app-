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