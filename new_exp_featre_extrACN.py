# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 05:04:43 2017

@author: vishh
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 05:03:23 2017

@author: vishh
"""

#this program is for implementing with scikit lear

import csv
import re
import collections
from nltk.metrics import scores
import nltk.classify 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics




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


def find_features(tweet):
  words = set(tweet)
  feature_arr = {}
  for w in featureList:
    feature_arr[w]= w in tweet
  return feature_arr



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
array = []
sent_lst = []

for row in inp_tweets:
#    print row
    #exit()
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))
    if sentiment == 'positive':
    	tmp=0
    if sentiment == 'negative':
    	tmp=1
    if sentiment == 'neutral':
    	tmp=2
  	sent_lst.append(str(tmp))


       
#    print tweets
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))
#print featureList
length = len(inp_tweets)
count =0
'''
featuresets = []
for sentiment,tweet in inp_tweets:
  featuresets.append(find_features(tweet),sentiment)
  print ((count*.1)/length)*100
  count +=1
'''

#featuresets = [(find_features(tweet),sentiment) for (sentiment,tweet) in inp_tweets]

index1 = 0
feature_arr = []

#fp = open('featureset.txt','w')

index1 = 0
temp_list = []
cntext_feature_vector = {}

for sentiment,tweet in inp_tweets:
  temp = tweet
  
  index2 = 0
  for word in temp:
    if word in featureList:
      if word in temp:
      	tmp = 1
    else:
      	tmp = 0
    temp_list.append(str(tmp))
  feature_arr.extend(temp_list)

'''
for i in range(len(inp_tweets))
  for j in range(len(inp_tweets[i][1]))
  print inp_tweets[]
np.set_printoptions(threshold=np.inf)
X,y = np.array(feature_arr).reshape(9225,6942)
sent_arr = np.array(sent_lst)


'''
'''
print "features number:",len(featureList)
print "Total tweets number:",len(inp_tweets)
  
print "Total length of the new list:",len(feature_arr)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_senti_train,X_senti_test = train_test_split(X, test_size=0.33, random_state=42)

print "X train size:",len(X_train)
print "X test size:",len(X_test)

print "y train size:",len(y_train)
print "y test size:",len(y_test)

tfidf_transformer = TfidfTransformer(use_idf=False).fit(X_train)
X_train_tf = tf_transformer.transform(X_train)
print "tfidf train shape:",X_train_tf,shape
clf = MultinomialNB().fit(X_train_idf,sent_arr)
cld_normal = MultinomialNB().fit(X_train,sent_arr)
X_test_tf = tf_transformer.transform(X_test)

predicted = clf.predict(X_test_tf)
target_names = ['positive', 'negative', 'neutral']

print(metrics.classification_report(X_test,predicted,target_names))

#np.savetxt('featureset.txt',array)

'''

'''
with open('features.txt','wb') as f:
  np.savetxt(f,array)
'''


'''
for row in range(len(array)):
  for column in range(len(arra[row][]):
    fp.write(array[row][column])
  fp.write("\n")
fp.closed
'''
