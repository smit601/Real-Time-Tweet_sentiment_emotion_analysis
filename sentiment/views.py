
from django.http import JsonResponse
from django.shortcuts import render, redirect, HttpResponse
from .forms import Sentiment_Typed_Tweet_analyse_form
from .sentiment_analysis_code import sentiment_analysis_code
from .forms import Sentiment_Imported_Tweet_analyse_form
from .tweepy_sentiment import Import_tweet_sentiment
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
def sentiment_analysis(request):
    return render(request, 'home/sentiment.html')

def sentiment_analysis_type(request):
    if request.method == 'POST':
        form = Sentiment_Typed_Tweet_analyse_form(request.POST)
        analyse = sentiment_analysis_code()
        if form.is_valid():
            tweet = form.cleaned_data['sentiment_typed_tweet']
            sentiment = analyse.predict_sentiment(tweet)
            args = {'tweet':tweet, 'sentiment':sentiment}
            return render(request, 'home/sentiment_type_result.html', args)

    else:
        form = Sentiment_Typed_Tweet_analyse_form()
        return render(request, 'home/sentiment_type.html')


import json
def sentiment_analysis_import(request):
    if request.method == 'POST':
        form = Sentiment_Imported_Tweet_analyse_form(request.POST)
        tweet_text = Import_tweet_sentiment()
        
        #analyse = sentiment_analysis_code()

        if form.is_valid():
            handle = form.cleaned_data['sentiment_imported_tweet']

            if handle[0] == '#':
                #df = pd.read_csv("#BJP_tweets.csv")
                df = tweet_text.get_hashtag(handle)  # Call get_hashtag function
                
                sentiment_counts = df['sentiment'].value_counts().to_dict()  # Sentiment distribution data
                labels = list(sentiment_counts.keys())
                values = list(sentiment_counts.values())
                list_of_tweets_and_sentiments = list(zip(df['text'], df['sentiment']))  # Tweet data for table
            

                # Prepare sentiment data for new charts
                df['created_at_datetime'] = pd.to_datetime(df['created_at_datetime'])
                # Extract date from 'created_at_datetime' column
                df['date'] = df['created_at_datetime'].dt.date
                # Group data by date and sentiment, and count the occurrences
                sentiment_over_time = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            
                # Reset index to make 'date' a regular column
                sentiment_over_time = sentiment_over_time.reset_index()
                # Prepare data for chart rendering
                date_labels = sentiment_over_time['date'].astype(str).tolist()
                sentiment_values = {
                    'positive': sentiment_over_time['positive'].tolist(),
                    'negative': sentiment_over_time['negative'].tolist(),
                    'neutral': sentiment_over_time['neutral'].tolist()
                }

                #for line chart percentage
                # Group data by date and sentiment, and count the occurrences
                sentiment_over_time = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

                # Calculate the total number of tweets per day
                total_tweets_per_day = sentiment_over_time.sum(axis=1)

                # Calculate the percentage of each sentiment per day
                sentiment_over_time_percentage = (sentiment_over_time.div(total_tweets_per_day, axis=0) * 100).fillna(0)
                sentiment_over_time_percentage = sentiment_over_time_percentage.round(0).fillna(0)
                # Prepare data for chart rendering
                date_labels = sentiment_over_time_percentage.index.astype(str).tolist()
                sentiment_percentages = {
                    'positive': sentiment_over_time_percentage['positive'].tolist(),
                    'negative': sentiment_over_time_percentage['negative'].tolist(),
                    'neutral': sentiment_over_time_percentage['neutral'].tolist()
                }
                
                # for cumulative area chart

                # Group data by date and sentiment, and calculate cumulative counts
                sentiment_over_time_cumulative = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0).cumsum()

                # Prepare data for chart rendering
                date_labels = sentiment_over_time_cumulative.index.astype(str).tolist()
                sentiment_cumulative_counts = {
                    'positive': sentiment_over_time_cumulative['positive'].tolist(),
                    'negative': sentiment_over_time_cumulative['negative'].tolist(),
                    'neutral': sentiment_over_time_cumulative['neutral'].tolist()
                }
                overall_tweets = df['clean_text']

                # Apply preprocessing to sentiment-specific tweets
                positive_tweets = df[df['sentiment'] == 'positive']['clean_text']
                negative_tweets = df[df['sentiment'] == 'negative']['clean_text']
                neutral_tweets = df[df['sentiment'] == 'neutral']['clean_text']
                def generate_and_save_word_cloud(sentiment_tweets, filename,title):
                    """Generates a word cloud from the provided text, saves it to a file,
                    and displays it using matplotlib (optional).

                    Args:
                        sentiment_tweets: A list of preprocessed tweets.
                        filename: The filename to save the word cloud image (e.g., 'overall_wordcloud.png').
                        title: The title for the word cloud (displayed in the plot).
                    """
                    text = ' '.join(sentiment_tweets)
                    
                    wordcloud = WordCloud(width=600, height=400, background_color='white').generate(text)

                    # Save the word cloud image
                    
                    wordcloud.to_file(f'D:\sentiment_emotion_analysis\sentiment\static\wordcloud\{filename}')

                    # Optional: Display the word cloud using matplotlib (comment out if not needed)
                    # plt.figure(figsize=(10, 5))
                    # plt.imshow(wordcloud, interpolation='bilinear')
                    #   
                    # plt.axis('off')  # Hide axis
                    # plt.show()

                # Generate and save word clouds for each sentiment category
                generate_and_save_word_cloud(overall_tweets, 'overall_wordcloud.png', 'Overall Tweets Word Cloud')
                generate_and_save_word_cloud(positive_tweets, 'positive_wordcloud.png', 'Positive Sentiment Word Cloud')
                generate_and_save_word_cloud(negative_tweets, 'negative_wordcloud.png', 'Negative Sentiment Word Cloud')
                generate_and_save_word_cloud(neutral_tweets, 'neutral_wordcloud.png', 'Neutral Sentiment Word Cloud')
                hashtags_flat = []
                for tags in df['hashtags']:
                    # Convert string representation of list to actual list of hashtags
                    
                    # Clean each hashtag and append to flattened list
                    hashtags_flat.extend([tag.strip().lower() for tag in tags])

                #hashtags_flat = [tag for sublist in df['hashtags'] for tag in sublist]
                hashtag_counts = pd.Series(hashtags_flat).value_counts().head(10)  # Top 10 hashtags

                hashtag_data = hashtag_counts.to_dict()
                labels3 = list(hashtag_data.keys())
                values3 = list(hashtag_data.values())
                
                #s4_1
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

                
                engagement_over_time = df.groupby('date')[['favorite_count', 'retweet_count', 'reply_count']].sum().reset_index()
                engagement_json = engagement_over_time.to_json(orient='records')
                #s3 chart2
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
                #s52
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
                    'handle':handle,
                    'date_labels': json.dumps(date_labels),
                    'sentiment_percentages': json.dumps(sentiment_percentages),
                    'sentiment_cumulative_counts': json.dumps(sentiment_cumulative_counts),
                    'sentiment_values': json.dumps(sentiment_values),
                    'values': values,
                    'df':df,
                    'sentiment_counts': sentiment_counts,
                    'list_of_tweets_and_sentiments': list_of_tweets_and_sentiments,
                
                    "hashtag_labels": labels3,
                    "hashtag_values": values3,
                    #s3_2
                    'top_words': top_words,
                    'top_frequencies': top_frequencies,
                    #s3_3
                    #s4_1
                    'tweet_lengths': bins,
                    'tweet_counts': counts,
                    
                    'engagement_data': engagement_json,

                    'sentiment_data': sentiment_json,
                    'bin': bin,
                    'count': count,
                    'mean_favorite_data': json.dumps(mean_favorite_data),
                    'mean_retweet_data': json.dumps(mean_retweet_data),
                    'mean_reply_data': json.dumps(mean_reply_data)
                }

                    
                  
                return render(request, 'home/sentiment_import_result_hashtag.html', context)

            if handle[0] == '@':
                handle = handle[1:]
                
                df = tweet_text.get_tweets(handle)  # Call get_tweets function
                # Access sentiment information directly from the DataFrame (assuming a 'sentiment' column)
                list_of_tweets_and_sentiments = list(zip(df['text'], df['sentiment']))  # Create list of tuples
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
                    
                    wordcloud.to_file(f'D:\sentiment_emotion_analysis\sentiment\static\wordcloud2\{filename}')

                    
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
                 
                    'list_of_tweets_and_sentiments': list_of_tweets_and_sentiments,
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
                return render(request, 'home/sentiment_import_result.html', context)
    else:
        form = Sentiment_Imported_Tweet_analyse_form()
        return render(request, 'home/sentiment_import.html')


