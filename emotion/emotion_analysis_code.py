


import re
from nltk.stem.wordnet import WordNetLemmatizer 
import itertools
import numpy as np
import nltk

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import urllib
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt



# Load model and tokenizer (already done in provided code)
task = 'emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]


class emotion_analysis_code():

  def preprocess(self, text):
    new_text = []
    for t in text.split():
        if isinstance(t, (int, float)):  # Check if the token is numeric
            t = str(t)  # Convert numeric token to string
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = '' if t.startswith('http') else t
        new_text.append(t.lower())  # Convert to lowercase
    return " ".join(new_text)

  # Apply model to each text and add sentiment results
  def predict_sentiment(self, tweet):
    tweet = self.preprocess(tweet)
    # Call the defined preprocess function
    encoded_input = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    predicted_sentiment = labels[ranking[0]]  # Get most probable sentiment]
    #df['sentiment']=predicted_sentiment
    #df.to_csv(f'{query}_tweets.csv', index=False)
    return predicted_sentiment

  