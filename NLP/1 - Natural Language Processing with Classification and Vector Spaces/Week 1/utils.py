import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


def process_tweet(tweet):
    """
    Función de proceso de tweet.
     Aporte:
         tweet: una cadena que contiene un tweet
     Producción:
         tweets_clean: una lista de palabras que contienen el tweet procesado

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Construye frecuencias.
     Aporte:
         tweets: una lista de tweets
         ys: una matriz de m x 1 con la etiqueta de sentimiento de cada tweet
             (ya sea 0 o 1)
     Producción:
         freqs: un diccionario que asigna cada par (palabra, sentimiento) a su
         frecuencia
    """
    # Convierta la matriz np en una lista, ya que zip necesita un archivo iterable.
    # Es necesario apretar o la lista termina con un elemento.
    # También tenga en cuenta que esto es solo un NOP si ys ya es una lista.
    yslist = np.squeeze(ys).tolist()

    # Comience con un diccionario vacío y complételo recorriendo todos los tweets
    # y sobre todas las palabras procesadas en cada tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs