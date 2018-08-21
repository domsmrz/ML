import pickle
import operator
import numpy as np
import random

class FeatureGenerator(object):
    def fit(self, data_dict=None):
        pass

    def transform(self, data_dict):
        new_features = {
            'poi_to_ratio': (operator.truediv, ('from_poi_to_this_person', 'to_messages')),
            'poi_from_ratio': (operator.truediv, ('from_this_person_to_poi', 'from_messages')),
        }

        for person_id, person_data in data_dict.items():
            for feature_name, feature_description in new_features.items():
                arguments = [person_data[k] for k in feature_description[1]]
                if any(x == 'NaN' for x in arguments):
                    feature_value = 'NaN'
                else:
                    feature_value = feature_description[0](*arguments)
                data_dict[person_id][feature_name] = feature_value
        return data_dict


class PopulateAndGuessMissing(object):
    def __init__(self):
        self.stds = None
        self.means = None

    def fit(self, data_dict):
        omit = {'email_address'}
        omit_and_poi = omit | {'poi'}
        tracker = {k: ([], []) for k in next(iter(data_dict.values())).keys() if k not in omit_and_poi}
        for person_dict in data_dict.values():
            for k in tracker.keys():
                v = person_dict[k]
                if v != 'NaN':
                    tracker[k][person_dict['poi']].append(v)

        for k, (v_npoi, v_poi) in tracker.items():
            if len(v_npoi) < 5 or len(v_poi) < 5:
                del tracker[k]

        print(sorted(tracker.keys()))

        for k, v in tracker.items():
            self.means[k] = tuple(np.mean(x) for x in v)
            self.stds[k] = tuple(np.std(x) for x in v)

    def transform(self, data_dict):
        random.seed(5)
        data = []
        number_of_npoi_clones = 100
        number_of_poi_clones = 100
        miss_probability = 0.3
        for person_id, person_data in data_dict:
            number_of_clones = number_of_poi_clones if person_data['poi'] else number_of_npoi_clones
            for _ in number_of_clones:
                pass


with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
del data_dict['TOTAL']

working_dict = dict()
validation_dict = dict()

random.seed(42)

validation_names = set(random.sample(data_dict.keys(), len(data_dict) / 5))
for k, v in data_dict.items():
    this_dict = validation_dict if k in validation_names else working_dict
    this_dict[k] = v

#print(sum(v['poi'] for v in validation_dict.values()))

clf = FeatureGenerator()
data_dict = clf.transform(data_dict)
clf = PopulateAndGuessMissing()
clf.fit(data_dict)
