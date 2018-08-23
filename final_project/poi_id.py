#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

class UnknownGuesser(object):
    def __init__(self):
        self.poi_means = None
        self.poi_stds = None
        self.no_poi_means = None
        self.no_poi_stds = None

        self.miss_prob = 0.3
        #self.poi_clones = 300
        #self.no_poi_clones = 100
        self.poi_clones = 500
        self.no_poi_clones = 1

    def fit(self, X, y):
        pois = X[y]
        no_pois = X[np.logical_not(y)]

        self.poi_means = np.nanmean(pois, axis=0)
        self.poi_stds = np.nanstd(pois, axis=0)

        self.no_poi_means = np.nanmean(no_pois, axis=0)
        self.no_poi_stds = np.nanstd(no_pois, axis=0)

    def transform_with_labels(self, X, y):
        new_features = []
        new_labels = []
        for row, label in zip(X, y):
            means = self.poi_means if label else self.no_poi_means
            miss_means = self.poi_means if not label else self.no_poi_means
            stds = self.poi_stds if label else self.no_poi_stds
            miss_stds = self.poi_stds if not label else self.no_poi_stds

            np.random.seed(42)
            for _ in range(self.poi_clones if label else self.no_poi_clones):
                this_features = []
                for feature, mean, miss_mean, std, miss_std in zip(row, means, miss_means, stds, miss_stds):
                    if np.isnan(feature):
                        miss = np.random.rand() < self.miss_prob
                        my_mean = miss_mean if miss else mean
                        my_std = miss_std if miss else std
                        this_features.append(np.random.normal(my_mean, my_std))
                    else:
                        this_features.append(feature)
                new_features.append(this_features)
                new_labels.append(label)
        return np.array(new_features, dtype=np.float), np.array(new_labels, dtype=np.bool)

    def transform_without_labels(self, X):
        new_features = []
        for row in X:
            for _ in range(10):
                data_point = []
                for value, poi_mean, poi_std, no_poi_mean, no_poi_std in zip(row, self.poi_means, self.poi_stds, self.no_poi_means, self.no_poi_stds):
                    if not np.isnan(value):
                        data_point.append(value)
                    else:
                        if True:
                        #if np.random.rand() < 0.9999:
                            # TODO
                            data_point.append(np.random.normal(poi_mean, poi_std))
                        else:
                            data_point.append(np.random.normal(no_poi_mean, no_poi_std))
                new_features.append(data_point)
            #new_features.append([x if not np.isnan(x) else (p_mean + np_mean) / 2 for x, p_mean, np_mean in zip(row, self.poi_means, self.no_poi_means)])
        return np.array(new_features, dtype=np.float)


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = [
    'poi',
    'bonus',
    'exercised_stock_options',
    'expenses',
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'other',
    'poi_from_ratio',
    'poi_to_ratio',
    'restricted_stock',
    'salary',
    'shared_receipt_with_poi',
    'to_messages',
    'total_payments',
    'total_stock_value',
]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#my_dataset = data_dict
my_dataset = data_dict.copy()
for k, v in my_dataset.items():
    my_dataset[k]['poi_to_ratio'] = float(v['from_poi_to_this_person']) / float(v['to_messages'])
    my_dataset[k]['poi_from_ratio'] = float(v['from_this_person_to_poi']) / float(v['from_messages'])

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
data_list = []
for _, person_dict in my_dataset.items():
    data_list.append([person_dict[k] for k in features_list])
data = np.array(data_list, dtype=np.float)
labels, features = targetFeatureSplit(data)
labels = np.array(labels, dtype=np.bool)

from sklearn import model_selection
working_features, validation_features, working_labels, validation_labels = \
    model_selection.train_test_split(features, labels, test_size=0.2, random_state=32)

working_features = np.array(working_features, dtype=np.float)
validation_features = np.array(validation_features, dtype=np.float)
working_labels = np.array(working_labels, dtype=np.bool)
validation_labels = np.array(validation_labels, dtype=np.bool)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

ug = UnknownGuesser()
ug.fit(working_features, working_labels)
f, l = ug.transform_with_labels(working_features, working_labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=22)
eps = 0.0000001
working_features = np.log2(np.abs(working_features) + eps)
for train_index, test_index in kf.split(working_features, working_labels):
    ug = UnknownGuesser()
    train_features, test_features = working_features[train_index], working_features[test_index]
    train_labels, test_labels = working_labels[train_index], working_labels[test_index]

    ug.fit(train_features, train_labels)
    train_features, train_labels = ug.transform_with_labels(train_features, train_labels)

    train_features = scale(train_features)

    pca = PCA(n_components=15)
    pca.fit(train_features)
    train_features = pca.transform(train_features)

    #lda = LinearDiscriminantAnalysis(n_components=2)
    #lda.fit(train_features, train_labels)
    #train_features = lda.transform(train_features)

    clf = DecisionTreeClassifier()
    #clf = SVC(probability=True)
    clf.fit(train_features, train_labels)


    # TODO: DELETE!!
    test_features = working_features[train_index]
    test_labels = working_labels[train_index]

    raw_pred = clf.predict(pca.transform(scale(ug.transform_without_labels(test_features))))
    #raw_pred = clf.predict_proba(pca.transform(scale(ug.transform_without_labels(test_features))))[:,1]
    pred = []
    for i in range(len(test_labels)):
        relevant = raw_pred[i*1000:(i+1)*1000]
        pred.append(round(sum(relevant)))

    pred = np.array(pred, dtype=np.bool)

    print("")
    print("Another fold")
    print("Accuracy: {}".format(sklearn.metrics.accuracy_score(test_labels, pred)))
    print("Recall: {}".format(sklearn.metrics.recall_score(test_labels, pred)))
    print("Precision: {}".format(sklearn.metrics.precision_score(test_labels, pred)))


    #points = pca.transform(scale(ug.transform_without_labels(test_features)))
    #plt.scatter(points[test_labels,0], points[test_labels,1], c='r')
    #plt.scatter(points[np.logical_not(test_labels),0], points[np.logical_not(test_labels),1], c='g')
    #plt.show()



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)