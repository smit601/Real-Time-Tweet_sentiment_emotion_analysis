
from django.http import JsonResponse
from django.shortcuts import render, redirect, HttpResponse
from .forms import Emotion_Typed_Tweet_analyse_form
from .emotion_analysis_code import emotion_analysis_code
from .forms import Emotion_Imported_Tweet_analyse_form
from .tweepy_emotion import Import_tweet_emotion
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict  
import datetime
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import datetime 
import json

def emotion_analysis(request):
    return render(request, 'home/emotion.html')

def emotion_analysis_type(request):
    if request.method == 'POST':
        form = Emotion_Typed_Tweet_analyse_form(request.POST)
        analyse = emotion_analysis_code()
        if form.is_valid():
            tweet = form.cleaned_data['emotion_typed_tweet']
            emotion = analyse.predict_sentiment(tweet)
            args = {'tweet':tweet, 'emotion':emotion}
            
            return render(request, 'home/emotion_type_result.html', args)

    else:
        form = Emotion_Typed_Tweet_analyse_form()
        return render(request, 'home/emotion_type.html')

def emotion_analysis_import(request):
    if request.method == 'POST':
        form = Emotion_Imported_Tweet_analyse_form(request.POST)
        tweet_text = Import_tweet_emotion()
        
        #analyse = emotion_analysis_code()

        if form.is_valid():
            handle = form.cleaned_data['emotion_imported_tweet']

            
            if handle[0] == '#':
                #df = pd.read_csv("#BJP_tweets.csv")
                df = tweet_text.get_hashtag(handle)  # Call get_hashtag function
                
                sentiment_counts = df['sentiment'].value_counts().to_dict()  # Sentiment distribution data
                labels = list(sentiment_counts.keys())
                values = list(sentiment_counts.values())
                list_of_tweets_and_emotions = list(zip(df['text'], df['sentiment']))  # Tweet data for table
                df['date'] = df['created_at_datetime'].dt.date
                overall_tweets = df['clean_text']
                df['tweet_length'] = df['text'].apply(len)
                tweet_lengths=list(df['tweet_length'])
                
                # Define histogram bins (ranges) and counts
                # Define bin sizes (e.g., 25 characters per bin)
                bin_size = 25

                # Create bins and initialize empty counts list
                bins = []
                counts = [0] * (max(tweet_lengths) // bin_size + 1)  # Pre-allocate counts list

                # Count tweets in each bin
                for length in tweet_lengths:
                    bin_index = length // bin_size  # Calculate bin index for the length
                    counts[bin_index] += 1  # Increment count for that bin

                # Adjust bins based on bin_size (e.g., 25-50, 51-75, etc.)
                for i in range(len(counts)):
                    lower_bound = i * bin_size
                    upper_bound = lower_bound + bin_size - 1  # Account for zero-based indexing
                    bins.append(f"{lower_bound}-{upper_bound}")  # Create bin labels

                print(bins,counts)
                
                def generate_and_save_word_cloud(sentiment_tweets, filename,title):
                   
                    text = ' '.join(sentiment_tweets)
                    
                    wordcloud = WordCloud(width=400, height=340, background_color='white').generate(text)

                    # Save the word cloud image
                    
                    wordcloud.to_file(f'D:\sentiment_emotion_analysis\emotion\static\emotion_wordcloud\{filename}')

                    
                generate_and_save_word_cloud(overall_tweets, 'overall_wordcloud.png', 'Overall Tweets Word Cloud')
                text_data = df['text']

                # Create a CountVectorizer to convert text data into a bag-of-words representation
                vectorizer = CountVectorizer(max_features=1000, stop_words='english')
                X = vectorizer.fit_transform(text_data)

                # Sum the occurrences of each word across all documents
                word_counts = X.sum(axis=0)

                # Get the feature names (words)
                words = vectorizer.get_feature_names_out()

                # Create a DataFrame for word frequencies
                word_frequencies = pd.DataFrame({'word': words, 'frequency': word_counts.flat})

                # Sort words by frequency in descending order
                word_frequencies = word_frequencies.sort_values(by='frequency', ascending=False)

                # Extract top N most frequent words and their frequencies
                top_words = word_frequencies['word'].iloc[:10].tolist()
                top_frequencies = word_frequencies['frequency'].iloc[:10].tolist()

                
                engagement_over_time = df.groupby('date')[['favorite_count', 'retweet_count', 'reply_count']].sum().reset_index()
                engagement_json = engagement_over_time.to_json(orient='records')
                df['total_engagements'] = df['favorite_count'] + df['retweet_count'] + df['reply_count']
                avg_engagement_by_sentiment = df.groupby('sentiment')['total_engagements'].mean().reset_index().round(0)

                # Convert DataFrame to JSON format
                sentiment_json = avg_engagement_by_sentiment.to_json(orient='records')

                histogram_data = df['total_engagements'].tolist()
                bin = []  # List to hold bin labels
                count = []  # List to hold bin counts

                # Calculate bins and counts based on engagement data
                maxEngagement = max(histogram_data)
                binSize1 = int(maxEngagement/5)
                binSize=round(binSize1,-2) # Define the bin size (e.g., 2000 engagements per bin)

                # Calculate counts for each bin
                for i in range(0, maxEngagement + binSize , binSize):
                    binLabel = f"{i}-{i + binSize - 1}"  # Create bin label (e.g., "0-1999")
                    bin.append(binLabel)  # Append bin label to list

                    # Calculate count of engagements falling into this bin range
                    count1 = sum(1 for eng in histogram_data if i <= eng < i + binSize)
                    count.append(count1)  # Append count to list

                mean_favorite_count = df.groupby('sentiment')['favorite_count'].mean().reset_index().round(0)
                mean_retweet_count = df.groupby('sentiment')['retweet_count'].mean().reset_index().round(0)
                mean_reply_count = df.groupby('sentiment')['reply_count'].mean().reset_index().round(0)

                # Convert DataFrames to dictionaries for JSON serialization
                mean_favorite_data = mean_favorite_count.to_dict('records')
                mean_retweet_data = mean_retweet_count.to_dict('records')
                mean_reply_data = mean_reply_count.to_dict('records')


                context = {
                    'labels': labels,
                    'values':values,
                    'handle':handle,
                 
                    'list_of_tweets_and_emotions': list_of_tweets_and_emotions,
                    'top_words': top_words,
                    'top_frequencies': top_frequencies,
                    'engagement_data': engagement_json,
                    'tweet_lengths': bins,
                    'tweet_counts': counts,
                    
                    'sentiment_data': sentiment_json,
                    'bin': bin,
                    'count': count,
                    'mean_favorite_data': json.dumps(mean_favorite_data),
                    'mean_retweet_data': json.dumps(mean_retweet_data),
                    'mean_reply_data': json.dumps(mean_reply_data),
                }
                  
                return render(request, 'home/emotion_import_result_hashtag.html', context)

            if handle[0] == '@':
                handle = handle[1:]
                
                df = tweet_text.get_tweets(handle)  # Call get_tweets function
                # Access sentiment information directly from the DataFrame (assuming a 'sentiment' column)
                list_of_tweets_and_emotions = list(zip(df['text'], df['sentiment']))  # Create list of tuples
                sentiment_counts = df['sentiment'].value_counts().to_dict()  # Sentiment distribution data

                labels = list(sentiment_counts.keys())
                values = list(sentiment_counts.values())
                df['date'] = df['created_at_datetime'].dt.date
                overall_tweets = df['clean_text']
                df['tweet_length'] = df['text'].apply(len)
                tweet_lengths=list(df['tweet_length'])
                
                # Define histogram bins (ranges) and counts
                # Define bin sizes (e.g., 25 characters per bin)
                bin_size = 25

                # Create bins and initialize empty counts list
                bins = []
                counts = [0] * (max(tweet_lengths) // bin_size + 1)  # Pre-allocate counts list

                # Count tweets in each bin
                for length in tweet_lengths:
                    bin_index = length // bin_size  # Calculate bin index for the length
                    counts[bin_index] += 1  # Increment count for that bin

                # Adjust bins based on bin_size (e.g., 25-50, 51-75, etc.)
                for i in range(len(counts)):
                    lower_bound = i * bin_size
                    upper_bound = lower_bound + bin_size - 1  # Account for zero-based indexing
                    bins.append(f"{lower_bound}-{upper_bound}")  # Create bin labels

                print(bins,counts)
                
                def generate_and_save_word_cloud(sentiment_tweets, filename,title):
                   
                    text = ' '.join(sentiment_tweets)
                    
                    wordcloud = WordCloud(width=400, height=340, background_color='white').generate(text)

                    # Save the word cloud image
                    
                    wordcloud.to_file(f'D:\sentiment_emotion_analysis\emotion\static\emotion_wordcloud2\{filename}')

                    
                generate_and_save_word_cloud(overall_tweets, 'overall_wordcloud.png', 'Overall Tweets Word Cloud')
                text_data = df['text']

                # Create a CountVectorizer to convert text data into a bag-of-words representation
                vectorizer = CountVectorizer(max_features=1000, stop_words='english')
                X = vectorizer.fit_transform(text_data)

                # Sum the occurrences of each word across all documents
                word_counts = X.sum(axis=0)

                # Get the feature names (words)
                words = vectorizer.get_feature_names_out()

                # Create a DataFrame for word frequencies
                word_frequencies = pd.DataFrame({'word': words, 'frequency': word_counts.flat})

                # Sort words by frequency in descending order
                word_frequencies = word_frequencies.sort_values(by='frequency', ascending=False)

                # Extract top N most frequent words and their frequencies
                top_words = word_frequencies['word'].iloc[:10].tolist()
                top_frequencies = word_frequencies['frequency'].iloc[:10].tolist()

                
                engagement_over_time = df.groupby('date')[['favorite_count', 'retweet_count', 'reply_count']].sum().reset_index()
                engagement_json = engagement_over_time.to_json(orient='records')
                df['total_engagements'] = df['favorite_count'] + df['retweet_count'] + df['reply_count']
                avg_engagement_by_sentiment = df.groupby('sentiment')['total_engagements'].mean().reset_index().round(0)

                # Convert DataFrame to JSON format
                sentiment_json = avg_engagement_by_sentiment.to_json(orient='records')

                histogram_data = df['total_engagements'].tolist()
                bin = []  # List to hold bin labels
                count = []  # List to hold bin counts

                # Calculate bins and counts based on engagement data
                maxEngagement = max(histogram_data)
                binSize1 = int(maxEngagement/5)
                binSize=round(binSize1,-2) # Define the bin size (e.g., 2000 engagements per bin)

                # Calculate counts for each bin
                for i in range(0, maxEngagement + binSize, binSize):
                    binLabel = f"{i}-{i + binSize - 1}"  # Create bin label (e.g., "0-1999")
                    bin.append(binLabel)  # Append bin label to list

                    # Calculate count of engagements falling into this bin range
                    count1 = sum(1 for eng in histogram_data if i <= eng < i + binSize)
                    count.append(count1)  # Append count to list

                mean_favorite_count = df.groupby('sentiment')['favorite_count'].mean().reset_index().round(0)
                mean_retweet_count = df.groupby('sentiment')['retweet_count'].mean().reset_index().round(0)
                mean_reply_count = df.groupby('sentiment')['reply_count'].mean().reset_index().round(0)

                # Convert DataFrames to dictionaries for JSON serialization
                mean_favorite_data = mean_favorite_count.to_dict('records')
                mean_retweet_data = mean_retweet_count.to_dict('records')
                mean_reply_data = mean_reply_count.to_dict('records')


                context = {
                    'labels': labels,
                    'values':values,
                    'handle':handle,
                 
                    'list_of_tweets_and_emotions': list_of_tweets_and_emotions,
                    'top_words': top_words,
                    'top_frequencies': top_frequencies,
                    'engagement_data': engagement_json,
                    'tweet_lengths': bins,
                    'tweet_counts': counts,
                    
                    'sentiment_data': sentiment_json,
                    'bin': bin,
                    'count': count,
                    'mean_favorite_data': json.dumps(mean_favorite_data),
                    'mean_retweet_data': json.dumps(mean_retweet_data),
                    'mean_reply_data': json.dumps(mean_reply_data),
                }
                return render(request, 'home/emotion_import_result.html', context)

    else:
        form = Emotion_Imported_Tweet_analyse_form()
        return render(request, 'home/emotion_import.html')