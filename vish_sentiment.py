import csv


def process_tweet(tweet):
	tweet = tweet.lower()
	return tweet
	
	tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
	tweet = re.sub(



with open('data/training_neatfile.csv','rb') as f:
	reader_csv = csv.reader(f)
	tweets_list = list(reader_csv)
for label,tweet in tweets_list:
	processed_tweet = process_tweet(tweet)
	print tweet
	print processed_tweet
	print "@@@@@@@@@@@@@@@@@@@@@@@@"
