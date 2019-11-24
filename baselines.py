
from creme.linear_model import LogisticRegression
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler
from creme.compose import Pipeline
from creme.metrics import Accuracy
from creme import compat
from creme import stream
from creme import datasets
import argparse
import time
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from skmultiflow.lazy import KNN
from skmultiflow.trees import HoeffdingTree
from creme import neighbors
from creme import naive_bayes
import numpy as np
import os
import psutil

process = psutil.Process(os.getpid())

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="BasELinE ModEL [1-8]")
ap.add_argument("-d", "--dataset", required=True,
    help="BasELinE ModEL [1-3]")
ap.add_argument("-w", "--window_size", required=True,
    help="Window Size [1-3]")
args = vars(ap.parse_args())

window_size = int(args['window_size'])

X_y = datasets.fetch_electricity()

types_1 = {"Elevation": float, "Aspect": float, "Slope": float ,"Horizontal_Distance_To_Hydrology": float,
         "Vertical_Distance_To_Hydrology": float, "Horizontal_Distance_To_Roadways": float,"Hillshade_9am": float,"Hillshade_Noon": float,
         "Hillshade_3pm": float, "Horizontal_Distance_To_Fire_Points": float,"Wilderness_Area1": float,"Wilderness_Area2": float,
         "Wilderness_Area3": float, "Wilderness_Area4": float,"Soil_Type1": float,"Soil_Type2": float,"Soil_Type3": float,"Soil_Type4": float,
         "Soil_Type5": float,"Soil_Type6": float,"Soil_Type7": float,"Soil_Type8": float,"Soil_Type9": float,"Soil_Type10": float,
         "Soil_Type11": float,"Soil_Type12": float,"Soil_Type13": float,"Soil_Type14": float,"Soil_Type15": float,"Soil_Type16": float,
         "Soil_Type17": float,"Soil_Type18": float,"Soil_Type19": float,"Soil_Type20": float,"Soil_Type21": float,"Soil_Type22": float,
         "Soil_Type23": float,"Soil_Type24": float,"Soil_Type25": float,"Soil_Type26": float,"Soil_Type27": float,"Soil_Type28": float,
         "Soil_Type29": float,"Soil_Type30": float,"Soil_Type31": float,"Soil_Type32": float,"Soil_Type33": float,"Soil_Type34": float,
         "Soil_Type35": float,"Soil_Type36": float,"Soil_Type37": float,"Soil_Type38": float,"Soil_Type39": float,"Soil_Type40": float
         ,"Cover_Type": int}

types_2 = {"Airline": int, "Flight": int, "AirportFrom": int, "AirportTo": int,
            "DayOfWeek": int, "Time": int, "Length": int, "Delay": int}

types_3 = {"date": float, "day": int, "period": float, "nswprice":float,
           "nswdemand": float, "vicprice": float, "vicdemand":float, "transfer": float, "class": int}

dataset_1 = stream.iter_csv('data/cover_types.csv', target_name="Cover_Type", converters = types_1)
dataset_2 = stream.iter_csv('data/airlines_numerical.csv', target_name="Delay", converters = types_2)
dataset_3 = stream.iter_csv('data/electricity_normalized_numerical.csv', target_name="class", converters = types_3)

datasets = [dataset_1, dataset_2, dataset_3]
dataset_index = int(args["dataset"]) - 1
dataset = datasets[dataset_index]

clf1 = MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(100, 50, ), warm_start = False, activation="relu")
clf2 = MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(100, 50, 50, 25), warm_start = False, activation="relu")
clf3 = KNN(n_neighbors=8, max_window_size=2000, leaf_size=40)
clf4 = GaussianNB()
clf5 = SGDClassifier(alpha=0.0001, average=False, class_weight=None,
       early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=1000,
       n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
       random_state=None, shuffle=True, tol=0.001, validation_fraction=0.1,
       verbose=0, warm_start=False)
clf6 = HoeffdingTree(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                      split_criterion='info_gain', split_confidence=1e-07, tie_threshold=0.05,
                       binary_split=False, stop_mem_management=False, remove_poor_atts=False,
                        no_preprune=False, leaf_prediction='nba', nb_threshold=0, nominal_attributes=None)
clf7 = PassiveAggressiveClassifier(C=1.0, fit_intercept=True, shuffle=False, verbose=0,
                                    loss='hinge', n_jobs=1, random_state=None, warm_start=False)

'''
    clf9 = GOOWE
'''
'''
Multinomial Naive Bayes for text classification
https://github.com/creme-ml/creme/blob/master/creme/naive_bayes/multinomial.py
Example in line 26
clf_text_nb = compose.Pipeline([('tokenize', feature_extraction.CountVectorizer(lowercase=False)), ('nb', naive_bayes.MultinomialNB(alpha=1))])
'''
classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]

classes_1 = np.array([1, 2, 3, 4, 5, 6, 7])
classes_2 = np.array([0, 1])
classes_3 = np.array([1, 0])
classes_all = [classes_1, classes_2, classes_3]

classes_s = classes_all[dataset_index]

classifier_index = int(args["model"]) - 1
model = classifiers[classifier_index]

metric = Accuracy()

acc = 0.0
total = 0.0
acc_num = 0.0
s_time = time.time()
x_list = []
y_list = []
counter = 0
if(classifier_index == 2):
    f_n_neighbors = 8
else:
    f_n_neighbors = 1
for (i, (X, y)) in enumerate(dataset):
    yy = np.array([y]).reshape(1, -1).ravel()
    xx = np.array([ v for v in X.values() ]).reshape(1, -1)
    if(i < f_n_neighbors):
        model.partial_fit(xx, yy, classes = classes_s)
    else:
        if counter < window_size:
            x_list.append(xx)
            y_list.append(yy)
            counter += 1
        else:
            x_array = np.asarray(x_list).reshape(window_size, -1)
            y_array = np.asarray(y_list).reshape(window_size,)
            preds = model.predict(x_array)
            acc_num += np.sum(y_array == preds)
            total = i+1
            acc = round((acc_num / total) * 100 , 2)
            model.partial_fit(x_array, y_array, classes = classes_s)
            elapsed_time = time.time() - s_time
            x_list = []
            y_list = []
            counter = 0
            memory_usage = process.memory_info().rss * 1e-6
            print("[INFO] update {} - Accuracy: {} - Elapsed Time: {} seconds - Memory Usage: {} MB".format(i, acc, round(elapsed_time), round(memory_usage, 2)))
e_time = time.time() - s_time
memory_usage = process.memory_info().rss * 1e-6
acc = round((acc_num / total) * 100 , 2)
print("[INFO] Final update - Accuracy: {} - Total Time: {} seconds - Memory Usage: {} MB".format(acc, round(e_time), round(memory_usage, 2)))
