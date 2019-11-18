
from creme.linear_model import LogisticRegression
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler
from creme.compose import Pipeline
from creme.metrics import Accuracy
from creme import compat
from creme import stream
import argparse
import time
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--csv", required=True,
    help="path to features CSV file")
ap.add_argument("-n", "--cols", type=int, required=True,
    help="# of feature columns in the CSV file (excluding class column")
args = vars(ap.parse_args())


#types = {"feat_{}".format(i): float for i in range(0, args["cols"])}
#types["class"] = int

types = {"Elevation": float, "Aspect": float, "Slope": float ,"Horizontal_Distance_To_Hydrology": float,
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

dataset = stream.iter_csv(args["csv"], target_name="Cover_Type", converters = types)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB()]

#model = compat.convert_sklearn_to_creme(estimator = linear_model.SGDClassifier(loss='log', eta0=0.01, learning_rate='constant'), classes=[False, True])
classifier_index = 1
model = compat.convert_sklearn_to_creme(estimator = classifiers[classifier_index], classes=[1, 2, 3, 4, 5, 6, 7])

model_ = Pipeline([
    ("scale", StandardScaler()),
    ("learn", OneVsRestClassifier(
        binary_classifier=LogisticRegression()))
 ])

metric = Accuracy()

s_time = time.time()
# loop over the dataset
for (i, (X, y)) in enumerate(dataset):
    # make predictions on the current set of features, train the
    # model on the features, and then update our metric
    preds = model.predict_one(X)
    model = model.fit_one(X, y)
    metric = metric.update(y, preds)
    elapsed_time = time.time() - s_time
    print("[INFO] update {} - {} - Elapsed Time: {} seconds".format(i, metric, round(elapsed_time)))

# show the accuracy of the model
e_time = time.time() - s_time()
print("[INFO] Final - {} - Total Time: {} seconds".format(metric, round(e_time)))
