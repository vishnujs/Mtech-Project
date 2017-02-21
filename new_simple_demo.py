import csv
import re
import collections
from nltk.metrics import scores
import nltk.classify 





'''
inpTweets = csv.reader(open('data/training_neatfile.csv', 'rb'), delimiter=',',
quotechar='|')

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
count =0

tweets=[]
for line in inpTweets:
#   tweets.append(line[1])
   print line

for line in inpTweets:
#   print line[0]
#   print line[1]
   print line
   count+=1
   print "count is",count
'''

#print "The length of the list is:",len(inpTweets)



#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL) 
    return pattern.sub(r"\1\1", s)
#end

#start process_tweet
def processTweet(tweet):
    # process the tweets
    
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)    
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
#    print "After Preprocessing:",tweet
    return tweet
#end 

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords
#end

#start getfeatureVector
def getFeatureVector(tweet, stopWords):
    featureVector = []  
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences 
        w = replaceTwoOrMore(w) 
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if it consists of only words
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
        #ignore if it is a stopWord
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
#	print "Feature Vector:",featureVector
    return featureVector    
#end

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
#    print "Features:",features
    return features
#end




with open('data/training_neatfile.csv','rb') as f:
   reader = csv.reader(f)
   inp_tweets = list(reader)
stopWords = getStopWordList('data/feature_list/stopwords.txt')

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
featureList = []
tweets = []

for label,line in inp_tweets:
	refsets[label].add(line)
#	print refsets
print "Executed the first phase"

'''
for line in refsets['positive']:
   print line
'''


'''
#the fowllowing line of code is used for finding the features for training the naive bayes classifier

featureVector = []  
words = tweet.split()
for w in words:
   #replace two or more with two occurrences 
   w = replaceTwoOrMore(w) 
   #strip punctuation
   w = w.strip('\'"?,.')
   #check if it consists of only words
   val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", w)
   #ignore if it is a stopWord
   if(w in stopWords or val is None):
      continue
   else:
      featureVector.append(w.lower())
'''
count1 = 0
count2 = 0

for row in inp_tweets:
#    print row
    #exit()
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))

       
#    print tweets
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))
#print "Feature LIst",featureList


training_set = nltk.classify.util.apply_features(extract_features, tweets)

NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

with open('data/training_neatfile_4.csv','rb') as f:
   reader_test = csv.reader(f)
   tweets_test = list(reader_test)
   
for label,line in tweets_test:
	testTweet = line
	processedTestTweet = processTweet(testTweet)
	observed = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
	testsets[observed].add(line)

#	print testsets 
print 'pos precision:', scores.precision(refsets['positive'], testsets['positive'])
print 'pos recall:', scores.recall(refsets['positive'], testsets['positive'])
print 'pos F-measure:', scores.f_measure(refsets['positive'], testsets['positive'])
   

