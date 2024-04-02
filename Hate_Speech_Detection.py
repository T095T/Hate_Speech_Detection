#!/usr/bin/env python
# coding: utf-8

# ## Hate Speech Recognition

# ** Problem Statement **
# 
#   Developing an automated hate speech detection system for online platforms to effectively identify and filter hate speech from user-generated content. The system should accurately differentiate between hate speech, offensive language, and benign content, contributing to a safer and more inclusive online environmen

# In[2]:


# IMporting libraries

from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[3]:


# Importing data
data = pd.read_csv('twitter.csv')
data.head(25)


# In[6]:


# Labellling the data

data['labels'] = data['class'].map({0:"Hate Speech" , 1:"Offensive Language" , 2:"No hate and offensive language"})
print(data.head())


# In[7]:


data = data[['tweet' , 'labels']]
print(data.head())


# ## NLP

# In[9]:


import re
import nltk
stemmer = nltk.SnowballStemmer('english')
from nltk.corpus import stopwords
import string
stopword = set(stopwords.words('english'))


# In[12]:


def cleantext(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]' , '' , text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]

    text = " ".join(text)
    return text
data["tweet"] = data["tweet"].apply(cleantext)
data.head(50)
    


# In[14]:


## Dividing data

x = np.array(data['tweet'])
y = np.array(data['labels'])


# In[16]:


cv =CountVectorizer()
X = cv.fit_transform(x) # Fitting the data in X
X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.3, random_state=42)



# In[18]:


clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# ### We get 87% accuracy

# In[19]:


def hate_recognition():
    import streamlit as st
    st.title("Hate Speech Detection")
    user = st.text_area("Enter your tweet:")
    if len(user)<1:
        st.write(" ")
    else:
        sample = user
        data= cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)

if __name__ == "__main__"   :     
    hate_recognition()


# In[20]:





# In[ ]:




