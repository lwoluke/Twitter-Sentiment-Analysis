# -*- coding: utf-8 -*-
"""
Twitter Sentiment Analysis using NLP and ML
- tweets are fetched then parsed using Python

@author: lwolu
"""

# Imported libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
import nltk
import emoji
import re
import matplotlib.pyplot as plt
from itertools import cycle, islice

def generate_tweets(posts, num):
    '''Show the most recent tweets from account'''
    count = 1
    
    print("MOST RECENT TWEETS: \n")
    
    for tweet in posts[0:num]:
        print("TWEET #" + str(count) + ": " + tweet.full_text + "\n")
        count += 1


def clean_text(txt):
    '''
    Clean the tweet text
    - Remove: 
        1. # Hash character (Not the whole hashtag)
        2. Hyperlinks
        3. Usernames
        4. Emojis
        5. RT (Retweets)
    '''
    txt = re.sub(r"RT[\s]+", "", txt)
    txt = txt.replace("\n", " ")
    txt = re.sub(" +", " ", txt)
    txt = re.sub(r"https?:\/\/\S+", "", txt)
    txt = re.sub(r"(@[A-Za-z0–9_]+)|[^\w\s]|#", "", txt)
    txt = emoji.replace_emoji(txt, replace='')
    txt = txt.strip()
    
    return txt

   
def get_subjectivity(txt):
    '''
    degree of opinion in tweet text [0, 1] where
    0 represents factual information and 1 represents personal opinion
    '''
    
    return TextBlob(txt).sentiment.subjectivity


def get_polarity(txt):
    '''
    floating point number [-1, 1] representing statements 
    where -1 is negative, 0 is neutral, and 1 is positive
    '''
    
    return TextBlob(txt).sentiment.polarity


def word_cloud(df):
    
    allWords = " ".join([tweets for tweets in df["Tweets"]])
    
    cloud = WordCloud(width = 900, height = 600, 
                          random_state = 16, max_font_size = 150).generate(allWords)
    plt.imshow(cloud, interpolation = "bilinear")
    plt.axis("off")
    
    plt.show()


def word_freq_bar_graph(df,column,title):
    '''
    Generates a bar graph of word frequencies in descending order
    '''
    
    topic_words = [ z.lower() for y in
                  [ x.split() for x in df[column] if isinstance(x, str)]
                  for z in y]
    
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])
    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))
    plt.title(title)
    
    plt.show()
    

def get_analysis(score):
    '''
    Determine if the tweet is negative, neutral, or positive
    Uses the polarity: floating point number [-1, 1]
    '''
    
    if score > 0:
        return "Positive"
    
    elif score == 0:
        return "Neutral"
    
    else:
        return "Negative"
    

def positive_tweets_only(df):
    ''' Print only positive tweets in ascending order '''
    
    print("ONLY POSITIVE TWEETS: \n")
    count = 1
    
    sortedDF = df.sort_values(by = ["Polarity"])
    for i in range(0, sortedDF.shape[0]):
        if sortedDF["Analysis"][i] == "Positive":
            print("TWEET #" + str(count) + ": " + sortedDF["Tweets"][i] + "\n")
        count += 1
        

def negative_tweets_only(df):
    '''Print only negative tweets in descending order'''
    
    print("ONLY NEGATIVE TWEETS: \n")
    count = 1
    
    sortedDF = df.sort_values(by = ["Polarity"], ascending = "False")
    for i in range(0, sortedDF.shape[0]):
        if sortedDF["Analysis"][i] == "Negative":
            print("TWEET #" + str(count) + ": " + sortedDF["Tweets"][i] + "\n")
        count += 1
        
    
def polarity_subjectivity_vis(df):
    '''Create polarity and subjectivity visualization'''
    
    plt.figure(figsize = (8, 6))
    for i in range(0, df.shape[0]):
        plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color = "Navy")
        
    plt.title("Sentiment Analysis")
    plt.xlabel("Polarity")
    plt.ylabel("Subjectivity")
    
    plt.show()


def pos_tweets_percentage(df):
    '''Get positive tweets percentage'''
    
    positiveTweets = df[df.Analysis == "Positive"]
    positiveTweets = positiveTweets["Tweets"]
    
    positiveTweets = round((positiveTweets.shape[0] / df.shape[0]) * 100, 2)
    
    print("% OF POSITIVE TWEETS: " + str(positiveTweets))


def neg_tweets_percentage(df):
    '''Get negative tweets percentage'''
    
    negativeTweets = df[df.Analysis == "Negative"]
    negativeTweets = negativeTweets["Tweets"]
    
    negativeTweets = round((negativeTweets.shape[0] / df.shape[0]) * 100, 2)
    
    print("% OF NEGATIVE TWEETS: " + str(negativeTweets))
    

def count_tweet_sentiment_vis(df):
    '''Create tweet count sentiment visualization'''
    plt.title("Sentiment Analysis", fontsize = 22)
    plt.xlabel("Sentiment", fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.ylabel("Counts", fontsize = 18)
    plt.yticks(fontsize = 14)
    
    my_colors = list(islice(['g', 'y', 'r'], None, 3))
    
    df["Analysis"].value_counts().plot(kind = "bar", stacked=True, color=my_colors)
    
    plt.show()
    

def main():
    '''
    Specify plot style 
    - Full list at https://www.dunderdata.com/blog/view-all-available-matplotlib-styles
    '''
    plt.style.use('seaborn')

    # Store csv file containing the Twitter API key information
    log = "C:/Users/lwolu/OneDrive/Documents"\
            "/Projects/TwitterAPI_Keys.csv"

    # Get specific credentials for the Twitter API 
    keys = open(log).read().splitlines()
         
    consumerKey = keys[0]
    consumerSecret = keys[1]
    accessToken = keys[2]
    accessTokenSecret = keys[3]

    # Set authentication and access token
    authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
    authenticate.set_access_token(accessToken, accessTokenSecret)

    # Create Twitter API using authentication information
    api = tweepy.API(authenticate, wait_on_rate_limit = True)

    # Extract n tweets from specified Twitter account
    posts = api.user_timeline(screen_name = "RobertDowneyJr", count = 250, tweet_mode = "extended")
    
    # Shows the most recent tweets
    generate_tweets(posts, 2)
        
    # Generate dataframe for tweets column
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=["Tweets"])
    
    # Cleaning tweet text
    df["Tweets"] = df["Tweets"].apply(clean_text)
    
    # Create two additional columns for subjectivity and polarity:  
    df["Subjectivity"] = df["Tweets"].apply(get_subjectivity)
    df["Polarity"] = df["Tweets"].apply(get_polarity)
    pd.set_option('display.max_columns', 4)
    
    # Visualize the most common words through
    word_cloud(df)
    
    # Visualize most frequent words in descending order
    plt.figure(figsize=(10,10))
    word_freq_bar_graph(df,"Tweets","Popular Words from User")
    
    # Create column determining if tweets are positive or negative
    df["Analysis"] = df["Polarity"].apply(get_analysis)
    
    # Print only positive tweets in ascending order
    positive_tweets_only(df)
    
    # Print only negative tweets in descending order
    negative_tweets_only(df)
    
    # Generates visualization for the relationship
    polarity_subjectivity_vis(df)
    
    # Get positve and negative %
    pos_tweets_percentage(df)
    neg_tweets_percentage(df)
    
    # Value counts
    df["Analysis"].value_counts()
    
    # Generates visualization for the relationship
    count_tweet_sentiment_vis(df)


if __name__ == "__main__":
    main()
