# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:43:59 2023

@author: RAJARSHI RAY
@Internship : CodSoft - Task - 3
"""

# Importing libraries and modules

import nltk
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset input

df = pd.read_csv('D:/CodSoft Internship/3. TASK_3 Email spam detection/spam.csv', encoding='ISO-8859-1')
df.columns

# Renaming v1 and v2 columns to type and message and dropping the other columns

df.rename(columns={'v1' : 'mail_type' , 'v2' : 'message'}, inplace = True)
df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],inplace=True)

df.head()

df.shape

# Dealing with null and duplicates

df.describe()  # dataset decription
df.info()      # dataset information

df.isnull().sum()   # null check
df.duplicated().sum()  # duplicate value check

df.drop_duplicates(keep='first', inplace=True)  # keeping only first unique and removing its duplicate

df.shape

# Dataset operations

df.mail_type.value_counts()  # checking number of data for each type

### Inference : dataset is imbalanced due to un equal distribution of values in categories
### Solution : we will stratify the target column in train test split

## Dictionary of target column numerical assign

mail_type_labels = {
    'ham' : 0 ,
    'spam' : 1
    }


## Mapping the dictionary to target column

df['mail_type'] = df['mail_type'].map(mail_type_labels)
df.head()


# Text processing using NLP

stemmer = PorterStemmer()

## Custom function to process data

def nlp_processing(df):
    
    # Final list to store processed words
    
    text_list = []
    
    for i in range(0,len(df)):
        
        try:
            # Regular expression to remove everything except alphabets
            
            text_processing = re.sub('[^a-zA-Z]',' ',df['message'][i])
            
            # convert to lowercase for simplicity
            
            text_processing = text_processing.lower()
            
            # split into words
            
            text_processing = text_processing.split()
            
            # stem the words for NLP which are not stopwords
            
            text_processing = [stemmer.stem(words) for words in text_processing if not words in stopwords.words('english')]
        
            # assemble the words
            
            text_processing = ' '.join(text_processing)
            
            # combine into list
            
            text_list.append(text_processing)
        
        except KeyError:
            continue
        
    return text_list
    

# call function on dataset

text_data_train_file = nlp_processing(df)

# Vectorizer to create numerical vectors of processed text

def vectorizer(text_processed_list):
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer()
    
    X_col = vectorizer.fit_transform(text_processed_list).toarray()
    
    return X_col

X_training_set = vectorizer(text_data_train_file)
X_training_set


# Separating the X and y

X = X_training_set
y = df['mail_type']

print(X.shape,y.shape)

# fix y









