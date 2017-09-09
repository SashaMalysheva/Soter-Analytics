import pickle
import os
import math
import numpy as np
np.random.seed(10)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


def find_noise(z, swing_height=0.2):
    """

    :param z: 1d array
        Measured value.
    :param swing_height: double, default = 0.2
        Range of noise.
    :return: 2d array
        Return array of peak points.

    """
    time = np.arange(len(z))
    points = []
    for i in range(1, len(z) - 1):
        if math.fabs(z[i - 1] - z[i]) - swing_height > 0 and math.fabs(z[i] - z[i + 1]) - swing_height > 0 and \
                                (z[i - 1] - z[i]) * (z[i] - z[i + 1]) < 0 and z[i] < 0:
            points.append([time[i], z[i]])
    return np.matrix(points).T


def find_raise_time(z):
    """
    :param z: 1d array
        Measured value.
    :return: int
        Raise time - time after bottom point.
    """
    time = np.arange(len(z))
    point = -1
    minim = 110001
    for i in range(0, len(z)):
        if z[i] < minim:
            point = i
            minim = z[i]
    return len(time) - point


def find_bottom_point(z):
    """

    :param z: 1d array
        Measured value.
    :return: int
        Value in bottom point.
    """
    minim = 110001
    for i in range(0, len(z)):
        if z[i] < minim:
            minim = z[i]
    return minim


def find_bottom_points(z, swing_height=0.2):
    """

    :param z: 1d array
        Measured value.
    :param swing_height: double, default = 0.2
        Range of noise.
    :return:
        Return the number of points close to to a minimum value.
    """
    minim = 110001
    for i in range(0, len(z)):
        if z[i] < minim:
            minim = z[i]
    ans = 0
    for i in range(0, len(z)):
        if z[i] - minim < swing_height:
            ans += 1
    return ans


def number_of_correct(y_p, y_t):
    """

    :param y_p: 1d array
        Predicted value.
    :param y_t: 1d array
        Test value.
    :return: int
        Number of correct values.
    """
    ans = 0
    for i in range(len(y_p)):
        if y_t[i] - 2 < y_p[i] < y_t[i] + 2:
            ans += 1
    return ans


files = [f for f in next(os.walk('.'))[2] if f.endswith('pickle')]

for f in files:
    with open(f, 'rb') as file:
        bends = pickle.load(file)
        x = np.zeros((11, len(bends)))
        y = np.zeros(len(bends))
        for i in range(len(bends)):
            x[5, i] = int(bends[i].axis_y_is_vertical)
            x[6, i] = bends[i].time_down
            x[7, i] = bends[i].time_static
            x[8, i] = bends[i].time_up
            x[9, i] = bends[i].bend_angle
            y[i] = bends[i].weight_up - bends[i].weight_down
        x = x.T
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8)

        names = ["Nearest Neighbors","Linear SVM", "Decision Tree",
                 "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes"]

        classifiers = [
            KNeighborsClassifier(),
            SVC(kernel="linear", C=0.025),
            DecisionTreeClassifier(),
            RandomForestClassifier(max_depth=5, n_estimators=10),
            MLPClassifier(solver="sgd", alpha=1),
            AdaBoostClassifier(learning_rate=0.5),
            GaussianNB()
        ]
        ans = np.zeros(len(names))
        i = 0

        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            ans[i] = clf.score(X_test, y_test)
            i += 1

        x = np.zeros((11, len(bends)))
        y = np.zeros(len(bends))
        for i in range(len(bends)):
            x[0, i] = find_noise(bends[i].z).shape[1] / len(bends[i].z)
            x[1, i] = find_raise_time(bends[i].z) / len(bends[i].z)
            x[2, i] = find_bottom_point(bends[i].z)
            x[3, i] = find_bottom_points(bends[i].z) / len(bends[i].z)
            x[4, i] = len(bends[i].z)
            x[5, i] = int(bends[i].axis_y_is_vertical)
            x[6, i] = bends[i].time_down
            x[7, i] = bends[i].time_static
            x[8, i] = bends[i].time_up
            x[9, i] = bends[i].bend_angle
            x[10, i] = bends[i].roll_angle
            y[i] = bends[i].weight_up - bends[i].weight_down
        x = x.T
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.8)
        i = 0
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            print(name, ans[i], " -> ", clf.score(X_test, y_test))
            i += 1
