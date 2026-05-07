# ===============================
# Spam Email Detection System
# Implemented using Naïve Bayes and TF-IDF
# ===============================

# Import required libraries
import numpy as np
import pandas as pd
import pickle
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===============================
# Load multiple datasets
# ===============================
df1 = pd.read_csv("spam_and_ham_classification.csv")
df2 = pd.read_csv("emails.csv")
df3 = pd.read_csv("mail_data.csv")
df4 = pd.read_csv("spam.csv")

# ===============================
# Standardize column names and label encoding
# ===============================
# Dataset 1
df1 = df1.rename(columns={'label': 'label', 'text': 'message'})
df1['label'] = df1['label'].map({'ham': 0, 'spam': 1})

# Dataset 2
df2 = df2.rename(columns={'text': 'message', 'spam': 'label'})

# Dataset 3
df3 = df3.rename(columns={'Category': 'label', 'Message': 'message'})
df3['label'] = df3['label'].map({'ham': 0, 'spam': 1})

# Dataset 4
df4 = df4.rename(columns={'v1': 'label', 'v2': 'message'})

df4['label'] = df4['label'].map({'safe': 0, 'spam': 1})

# Keep only required columns: message and label
df1 = df1[['message', 'label']]
df2 = df2[['message', 'label']]
df3 = df3[['message', 'label']]
df4 = df4[['message', 'label']]

# ===============================
# Combine all datasets into one
# ===============================
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# ===============================
# Feature Engineering: Subject extraction
# ===============================
# Take the first 7 words as the 'subject'
df['subject'] = df['message'].apply(lambda x: ' '.join(str(x).split()[:7]))
# Combine subject and message into a single feature
df['Email_Text'] = df['subject'] + " " + df['message']

# ===============================
# Text Cleaning
# ===============================
def clean_text(text):

    text = re.sub(r"http\S+|www\S+", " URL ", text)  # Replace URLs
    text = re.sub(r"\d+", " NUMBER ", text)          # Replace numbers
    return text

df['Email_Text'] = df['Email_Text'].apply(clean_text)

# ===============================
# Vectorization using TF-IDF
# ===============================
# Converts text into numerical features while considering word importance
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',       # Remove common English stopwords
    ngram_range=(1, 2),         # Consider unigrams and bigrams
    token_pattern=r'\b[a-zA-Z]{2,}\b',  # Include words with 2+ letters
    max_features=12000,         # Limit vocabulary size
    min_df=3                     # Ignore words that appear in less than 3 emails
)

X = vectorizer.fit_transform(df['Email_Text'])
y = df['label']

# ===============================
# Split dataset into train and test sets
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% data for testing
    random_state=42,
    stratify=y           # Maintain label distribution
)

# ===============================
# Train Naïve Bayes Classifier
# ===============================
model = MultinomialNB(alpha=0.3)  # Smoothing parameter
model.fit(X_train, y_train)

# ===============================
# Save the trained model and vectorizer
# ===============================
# Allows loading later without retraining
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

