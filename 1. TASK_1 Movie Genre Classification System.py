# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:07:47 2023

@author: Rajarshi Ray
"""
# Data input and textual data preprocessing

## 1. Importing Libraries 

import nltk
# nltk.download()

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

## 2. Dataset input

df = pd.read_csv('E:/CODSOFT INTERNSHIP/CODSOFT/1. TASK_1 Genre Classification Dataset/train_data.txt',sep=':::',names=['ID','TITLE','GENRE','DESCRIPTION'])
df.head()

## 3. Merge Title and Description columns

df['movie_details'] = df['TITLE'] + '' + df['DESCRIPTION']
df['movie_details']

## 4. View Column Names

df.columns

## 5. Textual data processing and cleaning with regular expression and NLTK

text_data = [] # in this list all the processed clean text data will be there

stemmer = PorterStemmer() # Declaring the stemmer object for text stemming
lemmatizer = WordNetLemmatizer() # Declaring the lemmatizer for text lemmatization

for i in range(0,len(df)):   # Iterating over the entire dataset 
    
    # RE to remove all other punctuations,numbers etc except the letters  on the combined column
    
    text_processing = re.sub('[^a-zA-Z]',' ',df['movie_details'][i])
    
    # Convert to lowercase for simplicity
    
    text_processing = text_processing.lower()
    
    # Split into words
    
    text_processing = text_processing.split()
    
    # Stem those words in the content which are not stopwords
    
    text_processing = [stemmer.stem(text_vals) for text_vals in text_processing if not text_vals in stopwords.words('english')]
    
    # Assemble the combined words processed
    
    text_processing = ' '.join(text_processing)
    
    # Put into the list
    
    text_data.append(text_processing)
    

## 6. View the processed text file

text_data
    
# Vectorizing the processed text data to numerical values using TD-IDF technique

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=2500)  # declaring the vectorizer object

X = vectorizer.fit_transform(text_data).toarray()  # convert the processed text to vectors

X

# Due..

# EDA on Genre data

df['GENRE'].value_counts()

# Encoding Genre to numerical

# Model building
















