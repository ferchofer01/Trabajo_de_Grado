import tweepy
import json
import os
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
import csv
import re #regular expression
from textblob import TextBlob
import preprocessor as p
import string

import pandas as pd
from csv import reader
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns



# 4 cadenas para la autenticacion
consumer_key = "hDNH6lOSagQRWR8RmMhY1UNbr"
consumer_secret = "wAwBUmf3PmxIEPnOcCXO29miffEVep0lIj7F6IkYCE9CZrKymV"
access_token = "1257878159187468288-lDwmojybRsroGX5XC4ljsjnwjRWUVs"
access_token_secret = "7SPeNfXJgX2fyG1hlIG5r6cO9ia9lQUbDnPg7bUii1InI"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)


# con este objeto realizaremos todas las llamadas al API
api = tweepy.API(auth,
                 wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)
				 
				 
# asi se realiza la declaracion de las ruta de los archivos

coronavirus_tweets = "datosAnalisis.csv"

# Estas serian las Columnas del archivo csv

#columns of the csv file
COLS = ['id', 'created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang',
        'favorite_count', 'retweet_count', 'original_author',   'possibly_sensitive', 'hashtags',
        'user_mentions', 'place', 'place_coord_boundaries']

#se declara el lapso de tiempo en el que quiero obtener mis datos
start_date = '2020-02-01'
end_date = '2020-05-01'

# Se realizara una declaracion de emoticones para poder asi ver la parte de sentimientos
#Emoticones de felicidad 
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

# Emoticones de Tristeza
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
#Patrones de emojin
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

# Combinacion de los Emoticones de Felicidad con los de Tristeza
emoticons = emoticons_happy.union(emoticons_sad)


#mrhod clean_tweets()
def clean_tweets(tweet):
    stop_words = set(stopwords.words('spanish'))
    word_tokens = word_tokenize(tweet)

    # después del preprocesamiento tweepy, el colon restante permanece después de eliminar las menciones
    # o signo RT al comienzo del tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    
    #reemplazar caracteres consecutivos no ASCII con un espacio
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

   
# eliminar emojis del tweet
    tweet = emoji_pattern.sub(r'', tweet)

    #filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []

    #looping through conditions
    for w in word_tokens:
        #check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
 

#method write_tweets()
def write_tweets(keyword, file):
    # If the file exists, then read the existing data from the CSV file.
    if os.path.exists(file):
        df = pd.read_csv(file, header=0)
    else:
        df = pd.DataFrame(columns=COLS)
    #page attribute in tweepy.cursor and iteration
    for page in tweepy.Cursor(api.search, q=keyword,
                              count=200, include_rts=False, since=start_date).pages(50):
        for status in page:
            new_entry = []
            status = status._json

            ## comprobar si los tweets estan en español o sino pasa al siguiente 
            if status['lang'] != 'es':
                continue

            #when run the code, below code replaces the retweet amount and
            #no of favorires that are changed since last download.
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                   status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue
            
            #tweepy preprocessing called for basic preprocessing
            clean_text = p.clean(status['text'])

            #call clean_tweet method for extra preprocessing
            filtered_tweet=clean_tweets(clean_text)

            #pass textBlob method for sentiment calculations
            blob = TextBlob(filtered_tweet)
            Sentiment = blob.sentiment

            #seperate polarity and subjectivity in to two variables
            polarity = Sentiment.polarity
			
            subjectivity = Sentiment.subjectivity

            #new entry append
            new_entry += [status['id'], status['created_at'],
                          status['source'], status['text'],filtered_tweet, Sentiment,polarity,subjectivity, status['lang'],
                          status['favorite_count'], status['retweet_count']]

            #to append original author of the tweet
            new_entry.append(status['user']['screen_name'])

            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None
            new_entry.append(is_sensitive)

            # hashtagas and mentiones are saved using comma separted
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)

            #get location of the tweet if possible
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)

            try:
                coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
            except TypeError:
                coordinates = None
            new_entry.append(coordinates)

            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)
            csvFile = open(file, 'a' ,encoding='utf-8')
            df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")


#declare keywords as a query for three categories
coronavirus_keywords = 'medidas en colombia OR coronavirus en colombia OR #coronavirus OR #covid-19 OR covid en colombia y el gobierno'

#call main method passing keywords and file path
df = write_tweets(coronavirus_keywords, coronavirus_tweets)

datos=pd.read_csv('datosAnalisis.csv')
df= pd.DataFrame(datos)

del df['id']
del df['created_at']
del df['source']
del df['lang']
del df['original_author']
del df['hashtags']
del df['user_mentions']
del df['place']
del df['place_coord_boundaries']
del df['original_text']
del df['clean_text']
del df['sentiment']
del df['possibly_sensitive']
del df['retweet_count']
del df['favorite_count']

print (df)

for i in df:
    df[i].replace("es", 1 , inplace=True)
    df[i].replace("polarity", -1 , inplace=True)
    df[i].replace("9", -1 , inplace=True)
    df[i].replace("subjectivity", 1 , inplace=True)
    df[i].replace("sentiment", -1 , inplace=True)
    df[i].replace(str, -1 , inplace=True)
    df[i].replace('0.03333333333333333', -1 , inplace=True)
    df[i].replace('0.2', -1 , inplace=True)
    df[i].replace('0', 1 , inplace=True)
    df[i].replace('0.0', 0 , inplace=True)
    df[i].replace('-0.1', -1 , inplace=True)
    df[i].replace('', -1 , inplace=True)
    df[i].replace('0.5', 1 , inplace=True)
df.dtypes


df.polarity = df.polarity.astype(float)
df.subjectivity = df.subjectivity.astype(float)

df.polarity = df.polarity.astype(int)
df.subjectivity = df.subjectivity.astype(int)

df.dtypes


#graficas

colors =("dodgerblue","salmon", "palevioletred", "steelblue","seagreen")
s=0
for col in df:
    sizes=df[col].value_counts()
    pie=df[col].value_counts().plot(kind='pie', colors=colors,shadow=True,autopct='%1.1f%%',
                                    startangle=30,radius=1.5,center=(0.5,0.5),
                                    textprops={'fontsize':12},frame=False,pctdistance=.65)
    labels=sizes.index.unique()
    plt.gca().axis("equal")
    plt.title(df.columns[s],weight='bold',size=14)
    plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)
    plt.savefig(str(df.columns[s])+'.png',dpi=100,bbox_inches="tight")
    s=s+1
    plt.show()


