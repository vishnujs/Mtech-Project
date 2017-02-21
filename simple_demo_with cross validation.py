# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 11:07:11 2017

@author: Admin CSE
"""


import csv
import re
import collections
from nltk.metrics import scores
import nltk.classify 
import time



start_time = time.clock()

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



with open('data/training_neatfile_cross_validation.csv','rb') as f:
   reader = csv.reader(f)
   inp_tweets = list(reader)
count_train = len(inp_tweets)
stopWords = getStopWordList('data/feature_list/stopwords.txt')

refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
featureList = []
tweets = []


	
#	print refsets
#print "Executed the first phase"

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
start_time = time.clock()
print "training data:",count_train
print "Extracting feature vectors"
count = 0
for row in inp_tweets:
#    print row
    #exit()
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))
    count += 1
    d=(count*1.0/count_train)*100
    print 'Extracting feature [%f%%]\r'%d

print "Extracting Features completed"
       
#    print tweets
#end loop

# Remove featureList duplicates
featureList = list(set(featureList))
#print "Feature LIst",featureList


training_set = nltk.classify.util.apply_features(extract_features, tweets)

NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print "Naive bayes Training completed"
count_test = 0
'''
with open('data/training_neatfile.csv','rb') as f:
   reader_test = csv.reader(f)
   tweets_test = list(reader_test)
count_test = len(tweets_test)
  ''' 
count = 0
print "Naive bayes tesing starting"
# for cross validation
initial_value=0
final_value=499
out_file = open('cross validation result.txt','a')
iteration = 0
while (final_value < count_train):
    count =0
    for label,line in inp_tweets[initial_value:final_value]:
        testTweet = line
        processedTestTweet = processTweet(testTweet)
        observed = NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet, stopWords)))
        testsets[observed].add(line)
        refsets[label].add(line)
        count += 1
        print count
        d=(count*1.0/500*100)
        print '[%d%%]Testing: [%f%%]\r'%(iteration,d)
        #	print testsets 
    precision = scores.precision(refsets['positive'], testsets['positive'])
    recall = scores.recall(refsets['positive'], testsets['positive'])
    fmeasure = scores.f_measure(refsets['positive'], testsets['positive'])
    print 'pos precision:', precision
    print 'pos recall:', recall
    print 'pos F-measure:', fmeasure
#        out_file.write("cross validation if [%d%%] - [%d%%] are:" % (initial_value,final_value)
#        outfile.write('precision:'+precison+'recall:'+recall+'fmeasure:'+fmeasure)
    out_file.write("Fold range"+str(initial_value)+"-"+str(final_value)+"\n")
    out_file.write("precision:"+str(precision))
    out_file.write("\trecall:"+str(recall))
    out_file.write("\tfmeasure:"+str(fmeasure))
    out_file.write("\n")
    initial_value = final_value
    final_value += 500
    iteration += 1
        
print("--- %s seconds ---" % (time.time() - start_time))     

