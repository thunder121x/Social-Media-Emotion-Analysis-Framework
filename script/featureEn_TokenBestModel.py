#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


# In[2]:


# Step 1: Load Data
df_train = pd.read_csv('../resource/Mental-Health-Twitter-Tokenized/train.csv')
df_val = pd.read_csv('../resource/Mental-Health-Twitter-Tokenized/val.csv')
df_test = pd.read_csv('../resource/Mental-Health-Twitter-Tokenized/test.csv')

# Prepare the features and target variable
X_train_text = df_train['processed_tokens']
X_val_text = df_val['processed_tokens']
X_test_text = df_test['processed_tokens']

y_train = df_train['label']
y_val = df_val['label']
y_test = df_test['label']


# In[ ]:


# Step 2: Text Vectorization (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)

# Convert processed tokens (list of tokens) to space-separated text
X_train_text = X_train_text.apply(lambda x: ' '.join(eval(x)))
X_val_text = X_val_text.apply(lambda x: ' '.join(eval(x)))
X_test_text = X_test_text.apply(lambda x: ' '.join(eval(x)))

# Transform the text data into TF-IDF features
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_val_tfidf = vectorizer.transform(X_val_text)
X_test_tfidf = vectorizer.transform(X_test_text)


# In[4]:


# Step 3: Numeric Features (followers, retweets, etc.)
numeric_features_train = df_train[['followers', 'friends', 'favourites', 'statuses', 'retweets', 'URLs', 'Mentions']]
numeric_features_val = df_val[['followers', 'friends', 'favourites', 'statuses', 'retweets', 'URLs', 'Mentions']]
numeric_features_test = df_test[['followers', 'friends', 'favourites', 'statuses', 'retweets', 'URLs', 'Mentions']]

# Standardize the numeric features
scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(numeric_features_train)
X_val_numeric = scaler.transform(numeric_features_val)
X_test_numeric = scaler.transform(numeric_features_test)

# Step 4: Combine Text and Numeric Features
X_train_combined = hstack([X_train_tfidf, X_train_numeric])
X_val_combined = hstack([X_val_tfidf, X_val_numeric])
X_test_combined = hstack([X_test_tfidf, X_test_numeric])


# In[ ]:


X_train_combined = X_train_tfidf
X_val_combined = X_val_tfidf
X_test_combined = X_test_tfidf


# In[5]:


# Step 5: Train the Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_combined, y_train)

# Step 6: Validation - Evaluate on the validation set
y_val_pred = model.predict(X_val_combined)
print("Validation Set Evaluation:")
print(classification_report(y_val, y_val_pred))

# # Step 7: After Validation, Train on the Full Training + Validation Data
# X_full_train = hstack([X_train_combined, X_val_combined])
# y_full_train = pd.concat([y_train, y_val], axis=0)

# # Re-train the model on the full training + validation data
# model.fit(X_full_train, y_full_train)


# In[6]:


# Step 8: Final Testing - Evaluate on the test set
y_test_pred = model.predict(X_test_combined)
print("Test Set Evaluation (Final Model):")
print(classification_report(y_test, y_test_pred))


# In[ ]:




