#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.cross_validation import  train_test_split
import numpy as np

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(train_features, train_labels)
pred = clf.predict(test_features)
print("Accuracy: {}".format(accuracy_score(test_labels, pred)))
print("Predicted positives: {}".format(sum(pred)))
print("Test size: {}".format(len(test_labels)))
print("True positives: {}".format(sum(np.logical_and(pred, test_labels))))
print("Precision: {}".format(precision_score(test_labels, pred)))
print("Recall: {}".format(recall_score(test_labels, pred)))
