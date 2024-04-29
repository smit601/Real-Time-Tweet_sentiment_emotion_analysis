

from .sentiment_analysis_code import sentiment_analysis_code
from twikit import Client
import pandas as pd
from IPython.display import display
import time
#from .sentiment_analysis_code import sentiment_analysis_code
class Import_tweet_sentiment:


    # Enter your account information (replace with placeholders)
    USERNAME = 'XXXXXXXXXX'
    EMAIL = 'XXXXXXXXXX'
    PASSWORD = 'XXXXXXXXXX'


    # Initialize the Twikit client
    client = Client('en-US')    

    # Check if saved cookies are available
    try:
    # Try to load cookies from a file
        client.load_cookies('cookies.json')
    except FileNotFoundError:
    # If cookies file is not found, perform login and save cookies
        client.login(
            auth_info_1=USERNAME,
            auth_info_2=EMAIL,
            password=PASSWORD
        )
    # Save the cookies for future use
    client.save_cookies('cookies.json')

    def get_tweets(self, query):
        
        tweet_data = []
        user = self.client.get_user_by_screen_name(query)
        desired_language='en'
        tweets = user.get_tweets('Tweets')
        for tweet in tweets:
            if tweet.lang == desired_language:
                tweet_info = {attr: getattr(tweet, attr) for attr in dir(tweet) if not attr.startswith('_') and not callable(getattr(tweet, attr))}
                tweet_info ['clean_text']=sentiment_analysis_code().preprocess(tweet.text)
                tweet_info['sentiment'] = sentiment_analysis_code().predict_sentiment(tweet.text)
                tweet_data.append(tweet_info)
          # Call sentiment analysis function
        df = pd.DataFrame(tweet_data)
        df.to_csv(f'{query}_tweets.csv', index=False)                                               
        return df
    
    def get_hashtag(self, query, count_per_request=20):
        # Initialize an empty list to store tweet data
        tweet_data = []
        seen_tweet_ids = set()

        desired_language='en'
        total_tweets_to_fetch=30

        # First search (using `search_tweet`)
        initial_tweets = self.client.search_tweet(query, 'Top', count_per_request)

        # Process initial tweets (optional)
        for tweet in initial_tweets:
            if tweet.lang == desired_language and tweet.id not in seen_tweet_ids:
                seen_tweet_ids.add(tweet.id)

                # Capture tweet information
                tweet_info = {attr: getattr(tweet, attr) for attr in dir(tweet) if not attr.startswith('_') and not callable(getattr(tweet, attr))}
                tweet_info ['clean_text']=sentiment_analysis_code().preprocess(tweet.text)
                tweet_info['sentiment'] = sentiment_analysis_code().predict_sentiment(tweet.text)  # Call sentiment analysis function
                tweet_data.append(tweet_info)   

        # Loop to fetch more tweets using `tweets.next()`
        current_tweets = initial_tweets
        while len(tweet_data) < total_tweets_to_fetch and current_tweets is not None:
            try:
                current_tweets = current_tweets.next()  # Attempt to retrieve more tweets
                if not current_tweets:  # Check if no more tweets are available
                    break

                for tweet in current_tweets:
                    if tweet.lang == desired_language and tweet.id not in seen_tweet_ids:
                        seen_tweet_ids.add(tweet.id)

                        # Capture tweet information
                        tweet_info = {attr: getattr(tweet, attr) for attr in dir(tweet) if not attr.startswith('_') and not callable(getattr(tweet, attr))}
                        tweet_info ['clean_text']=sentiment_analysis_code().preprocess(tweet.text)
                        tweet_info['sentiment'] = sentiment_analysis_code().predict_sentiment(tweet.text)  # Call sentiment analysis function
                        tweet_data.append(tweet_info)   
                        #all_tweets.append(tweet.text)
                      
            except StopIteration:  # Handle cases where no more tweets are available
                break

            time.sleep(2)  # Delay between requests
         # Save the DataFrame to a CSV file
             
        df = pd.DataFrame(tweet_data)  
        df.to_csv(f'{query}_tweets.csv', index=False)
        
        return df
