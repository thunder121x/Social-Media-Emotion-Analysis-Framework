#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv(r'../resource/Mental-Health-Twitter.csv')

# Show basic info
print(df.shape)
df.head()


# In[2]:


df = df.drop(columns=['Unnamed: 0'])


# In[3]:


# Check for missing values
print(df.isnull().sum())


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot label distribution
sns.countplot(x='label', data=df)
plt.title('Distribution of Labels (0 = Not Mental Health, 1 = Mental Health)')
plt.show()

# Optional: show exact numbers
print(df['label'].value_counts())


# In[5]:


# Check for completely duplicate rows
duplicate_rows = df[df.duplicated(keep=False)]
if not duplicate_rows.empty:
    print("Duplicate rows found:")
    print(duplicate_rows.shape)
else:
    print("No duplicate rows found.")


# # Preprocessing

# In[6]:


import pandas as pd
import re


# In[7]:


# Load the data
df = pd.read_csv(r'../Mental-Health-Twitter.csv')
df.head()


# In[8]:


df_cleaned = df.copy()


# In[9]:


df_cleaned = df_cleaned.drop(columns=['Unnamed: 0'])
df_cleaned = df_cleaned.drop_duplicates(keep=False)


# In[10]:


# Define a function to remove URLs and mark if there was a URL
def remove_urls(text):
    url_pattern = r'http\S+|www\S+'
    has_url = bool(re.search(url_pattern, text))  # Check if there is a URL
    cleaned_text = re.sub(url_pattern, '', text)  # Remove URLs
    return cleaned_text, has_url

# Apply the function to each row
df_cleaned[['post_text', 'URLs']] = df_cleaned['post_text'].apply(lambda x: pd.Series(remove_urls(x)))

# Done!
df_cleaned[['post_text', 'URLs']].head()


# In[11]:


# --- Step 2: Remove Mentions and track ---
def remove_mentions(text):
    mention_pattern = r'@\w+'
    has_mention = bool(re.search(mention_pattern, text))
    cleaned_text = re.sub(mention_pattern, '', text)
    return cleaned_text, has_mention

df_cleaned[['post_text', 'Mentions']] = df_cleaned['post_text'].apply(lambda x: pd.Series(remove_mentions(x)))
df_cleaned[['post_text', 'Mentions']].head()


# In[12]:


# --- Step 3: Handle Hashtags (extract and optionally remove) ---
def extract_hashtags(text):
    hashtags = re.findall(r'#\w+', text)  # Find hashtags
    return hashtags

# Extract hashtags into a new column
df_cleaned['Hashtags'] = df_cleaned['post_text'].apply(lambda x: extract_hashtags(x))

# (Optional) If you want to also REMOVE hashtags from the post_text:
def remove_hashtags(text):
    return re.sub(r'#\w+', '', text)

# Uncomment below line if you want to remove hashtags from the post_text
df_cleaned['post_text'] = df_cleaned['post_text'].apply(lambda x: remove_hashtags(x))

# View the result
df_cleaned[['post_text', 'Hashtags']].head()


# In[13]:


import emoji
import contractions


# In[14]:


# --- Step 4: Handle Emojis ---

# def convert_emojis(text):
#     return emoji.demojize(text, language='en')

def convert_emojis(text):
    # Convert emojis to :emoji_name:
    text = emoji.demojize(text, language='en')
    # Add space around emoji descriptions
    text = re.sub(r':([a-zA-Z_]+):', r' \1 ', text)
    # Replace underscores with spaces
    text = text.replace('_', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_cleaned['post_text'] = df_cleaned['post_text'].apply(lambda x: convert_emojis(x))

# --- Step 5: Expand Contractions ---
def expand_contractions(text):
    return contractions.fix(text)

df_cleaned['post_text'] = df_cleaned['post_text'].apply(lambda x: expand_contractions(x))

# --- Step 6: Remove Special Characters/Punctuation ---
def remove_special_characters(text):
    # Remove all characters except words, spaces, and hashtags
    cleaned_text = re.sub(r'[^\w\s#]', '', text)
    return cleaned_text

df_cleaned['post_text'] = df_cleaned['post_text'].apply(lambda x: remove_special_characters(x))

df_cleaned['post_text'] = df_cleaned['post_text'].str.replace(r'RT\s{2}', '', regex=True)


# In[15]:


# View the result
df_cleaned[['post_text', 'URLs', 'Mentions', 'Hashtags']].head()


# In[16]:


# Function to clean text (you can expand this based on your needs)
def clean_text(text):
    # Remove non-alphanumeric characters, convert to lowercase
    return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

# Add a cleaned version of the post text to the DataFrame
df_cleaned['post_text'] = df_cleaned['post_text'].apply(clean_text)
df_cleaned['post_text'].head()


# In[19]:


import json

with open('../resource/slang.json', 'r', encoding='utf-8') as f:
    slang_dict = json.load(f)

# slang_dict


# In[20]:


# Download standard English words (for filtering)
import nltk
from nltk.corpus import words
nltk.download('words')

standard_words = set(words.words())

def normalize_slang(text):
    word_list = text.split()
    normalized_words = []
    for word in word_list:
        if word.lower() not in standard_words:
            normalized_word = slang_dict.get(word.lower(), word)
            normalized_words.append(normalized_word)
        else:
            normalized_words.append(word)
    return ' '.join(normalized_words)

df_cleaned['post_text'] = df_cleaned['post_text'].apply(normalize_slang)

df_cleaned['post_text'] = df_cleaned['post_text'].apply(normalize_slang)


# In[21]:


# Save
label_col = df_cleaned.pop('label')

df_cleaned['label'] = label_col

df_cleaned.to_csv('../resource/Mental-Health-Twitter-Preprocessed.csv', index=False)


# In[ ]:




