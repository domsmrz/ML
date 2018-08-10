#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
start = time()
clf.fit(features_train, labels_train)
end = time()
print("training time: {:.3f}s".format(end-start))

start = time()
labels_predicted = clf.predict(features_test)
end = time()
print("prediction time: {:.3f}s".format(end-start))

print(accuracy_score(labels_predicted, labels_test))



#########################################################


