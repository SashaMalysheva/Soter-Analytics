import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import math


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


files = [f for f in next(os.walk('.'))[2] if f.endswith('pickle')]

for f in files:
    with open(f, 'rb') as file:
        bends = pickle.load(file)
        noises = np.zeros((2, len(bends)))
        raise_time = np.zeros((2, len(bends)))
        bottom_point = np.zeros((2, len(bends)))
        bottom_points = np.zeros((2, len(bends)))
        all_time = np.zeros((2, len(bends)))

        for i in range(len(bends)):
            noises[0, i] = find_noise(bends[i].z).shape[1] / len(bends[i].z)
            noises[1, i] = bends[i].weight_up - bends[i].weight_down
            raise_time[0, i] = find_raise_time(bends[i].z) / len(bends[i].z)
            raise_time[1, i] = bends[i].weight_up - bends[i].weight_down
            bottom_point[0, i] = find_bottom_point(bends[i].z)
            bottom_point[1, i] = bends[i].weight_up - bends[i].weight_down
            bottom_points[0, i] = find_bottom_points(bends[i].z) / len(bends[i].z)
            bottom_points[1, i] = bends[i].weight_up - bends[i].weight_down
            all_time[0, i] = len(bends[i].z)
            all_time[1, i] = bends[i].weight_up - bends[i].weight_down

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, sharex=True)
        # ax1.plot(noises[0], noises[1], '^g')
        # ax2.plot(raise_time[0], raise_time[1], '^r')
        # ax3.plot(bottom_point[0], bottom_point[1], '^b')
        # ax4.plot(bottom_points[0], bottom_points[1], '^g')
        ax5.plot(all_time[0], all_time[1], '^g')

        plt.show()
