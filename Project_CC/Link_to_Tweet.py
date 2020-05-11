# Import libraries
import tweepy
import threading, time
from datetime import datetime

# Keys
consumer_key = "DOuCN8WDK3wn2t4jZURScYldL"
consumer_secret = "mrxK51iBqOubaeRvlRlTmPTpDCdP5YiPmqvO7WLc7FISHPuH0l"
access_token = "1241367764750733313-AxuQhY63z0rtdGgrmDme6bLLiKH3ak"
access_token_key = "mxMqCnrNcIIuOBm5US59LYnC2jTYtjROe9MXF6FrxfjAY"

# Authorisation
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_key)
api = tweepy.API(auth)

# Print the poem
def print_poem():
    f = open("output.txt", "r", encoding='utf-8')
    poem = ''
    for i in range(5):
        poem += f.readline()
    api.update_status(status=poem)
    print("Post Successfully!")

def print_time():
    if datetime.now().minute == 0 and datetime.now().second == 0:
        if datetime.now().hour == 0:
            tweet = str("Bang " * 24)
        else:
            tweet = str("Bang " * datetime.now().hour)
        # api.update_status(status=tweet)
        print_poem()
        print("New hour!")
    time.sleep(1)
    print_time()




print_time()
