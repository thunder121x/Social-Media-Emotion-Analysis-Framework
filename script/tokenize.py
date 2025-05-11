#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv(r'../resource/Mental-Health-Twitter-Preprocessed.csv')
df.head()


# In[4]:


df['post_text'].head(15)


# In[5]:


import nltk
# nltk.download('punkt')  # Download tokenizer models

from nltk.tokenize import word_tokenize

# Tokenize the post_text column
df['tokens'] = df['post_text'].apply(lambda x: word_tokenize(x.lower()))
df['tokens']


# In[6]:


from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def remove_stopword(tokens):
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

df['processed_tokens'] = df['tokens'].apply(remove_stopword)

# Convert POS tag from nltk format to wordnet format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default

def lemmatize_tokens(tokens):
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]


df['processed_tokens'] = df['processed_tokens'].apply(lemmatize_tokens)
df[['tokens','processed_tokens']]


# In[7]:


# Save
label_col = df.pop('label')

df['label'] = label_col

df.to_csv('../resource/Mental-Health-Twitter-Tokenized.csv', index=False)


# In[ ]:




