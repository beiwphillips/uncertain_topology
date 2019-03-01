from glob import glob
import scipy.io
import sklearn.cluster
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import nglpy_cuda as ngl
import topopy

from utpy.test_functions import *

def load_data(foo="ackley", noise_level=0.3):
    assignment_map = {"4peaks": assignments_4peaks, "ackley": assignments_ackley, "salomon": assignments_salomon}
    base_name = "data/" + foo

    ground_truth = scipy.io.loadmat(base_name + "/groundtruth.mat")['gt']
    uncertain_realizations = scipy.io.loadmat("{}/{}_uncertain.mat".format(base_name, str(noise_level)))['noisyEnsemble']

    assignments = assignment_map[foo]
    return ground_truth, uncertain_realizations, assignments

def load_ensemble(name="matVelocity"):
    base_name = "data/" + name
    files = glob("{}/*.mat".format(base_name))

    uncertain_realizations = []
    for filename in files:
        token = filename.rsplit("/", 1)[1].split(".")[0]
        # Temperature data is a bit special in that it has multiple
        # entries, let's just take the first for now.
        if name == "matTemperature":
            uncertain_realizations.append(scipy.io.loadmat(filename)[token][:,:,0].T)
        else:
            uncertain_realizations.append(scipy.io.loadmat(filename)[token].T)
    return np.array(uncertain_realizations).T

def massage_data(grid):
    X = []
    Y = []
    for row, vals  in enumerate(grid):
        for col, val in enumerate(vals):
            X.append([col, row])
            Y.append(val)
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)

def alpha_blend(src, dest):
    out_image = np.zeros(dest.shape)
    out_image[:, :, 3] = src[:, :, 3] + dest[:, :, 3]*(1 - src[:, :, 3])
    out_image[:, :, :-1] = src[:,:,:-1]*src[:,:3] + dest[:,:,:-1]*dest[:,:,3]*(1 - src[:, :, 3])
    out_image[:, :, :-1] /= out_image[:, :, 3]

def overlay_alpha_image_lazy(background_rgb, overlay_rgba, alpha):
    # cf https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
    # If the destination background is opaque, then
    #   out_rgb = overlay_rgb * overlay_alpha + background_rgb * (1 - overlay_alpha)
    overlay_alpha = overlay_rgba[: , : , 3].astype(np.float) / 255. * alpha
    overlay_alpha_3 = np.dstack((overlay_alpha, overlay_alpha, overlay_alpha))
    overlay_rgb = overlay_rgba[: , : , : 3].astype(np.float)
    background_rgb_f = background_rgb.astype(np.float)
    out_rgb = overlay_rgb * overlay_alpha_3 + background_rgb_f * (1. - overlay_alpha_3)
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb

def overlay_alpha_image_precise(background_rgb, overlay_rgba, alpha, gamma_factor=2.2):
    """
    cf minute physics brilliant clip "Computer color is broken" : https://www.youtube.com/watch?v=LKnqECcg6Gw
    the RGB values are gamma-corrected by the sensor (in order to keep accuracy for lower luminancy),
    we need to undo this before averaging.
    """
    overlay_alpha = overlay_rgba[: , : , 3].astype(np.float) / 255. * alpha
    overlay_alpha_3 = np.dstack((overlay_alpha, overlay_alpha, overlay_alpha))

    overlay_rgb_squared = np.float_power(overlay_rgba[: , : , : 3].astype(np.float), gamma_factor)
    background_rgb_squared = np.float_power( background_rgb.astype(np.float), gamma_factor)
    out_rgb_squared = overlay_rgb_squared * overlay_alpha_3 + background_rgb_squared * (1. - overlay_alpha_3)
    out_rgb = np.float_power(out_rgb_squared, 1. / gamma_factor)
    out_rgb = out_rgb.astype(np.uint8)
    return out_rgb

def count_persistence(grid):
    X, Y = massage_data(grid)
    h, w = grid.shape

    graph = ngl.EmptyRegionGraph(index=None, max_neighbors=10, relaxed=False, beta=1, p=2.)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization='feature')
    tmc.build(X, Y)

    partitions = tmc.get_partitions()
    sorted_hierarchy = sorted([(p, k, x) for k, (p, x, s) in tmc.merge_sequence.items()])

    field = np.zeros(Y.shape, dtype=int)
    for k, v in partitions.items():
        field[v] = k

    counts = np.zeros(Y.shape)
    weighted_counts = np.zeros(Y.shape)
    last_persistence = 0
    for persistence, dying_index, surviving_index in sorted_hierarchy:
        next_field = np.array(field)
        next_field[np.where(next_field == dying_index)] = surviving_index

        counts[np.where(field == next_field)] += 1
        weighted_counts[np.where(field == next_field)] += (persistence-last_persistence)
        field = next_field
        last_persistence = persistence

    return counts.reshape(grid.shape), weighted_counts.reshape(grid.shape)

def assignments_4peaks(grid):
    X, Y = massage_data(grid)
    h, w = grid.shape

    graph = ngl.EmptyRegionGraph(index=None, max_neighbors=10, relaxed=False, beta=1, p=2.)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization='feature')
    tmc.build(X, Y)

    correct_p = 0
    for p in tmc.persistences:
        if len(tmc.get_partitions(p).keys()) == 4:
            correct_p = p
            partitions = tmc.get_partitions(p)

    field = np.zeros(Y.shape, dtype=int)
    for k, v in partitions.items():

        x = k // 41
        y = k % 41
        field[v] = 0
        if x > 23 and y > 23:
            field[v] = 3
        elif y > 23:
            field[v] = 2
        elif x > 23:
            field[v] = 1

    return field.reshape(grid.shape), correct_p

def assignments_ackley(grid):
    X, Y = massage_data(grid)
    h, w = grid.shape

    graph = ngl.EmptyRegionGraph(index=None, max_neighbors=10, relaxed=False, beta=1, p=2.)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization='feature')
    tmc.build(X, Y)

    correct_p = 0
    for p in tmc.persistences:
        if len(tmc.get_partitions(p).keys()) == 9:
            correct_p = p
            partitions = tmc.get_partitions(p)

    field = np.zeros(Y.shape, dtype=int)
    for k, v in partitions.items():

        x = k // 41
        y = k % 41
        field[v] = 0
        if x > 31 and y > 31:
            field[v] = 8
        elif x > 31 and y > 9:
            field[v] = 7
        elif x > 31:
            field[v] = 6
        elif x > 9 and y > 31:
            field[v] = 5
        elif x > 9 and y > 9:
            field[v] = 4
        elif x > 9:
            field[v] = 3
        elif y > 31:
            field[v] = 2
        elif y > 9:
            field[v] = 1
        else:
            field[v] = 0

    return field.reshape(grid.shape), correct_p

def assignments_salomon(grid):
    X, Y = massage_data(grid)
    h, w = grid.shape

    graph = ngl.EmptyRegionGraph(index=None, max_neighbors=10, relaxed=False, beta=1, p=2.)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization='feature')
    tmc.build(X, Y)

    correct_p = 0
    for p in tmc.persistences:
        if len(tmc.get_partitions(p).keys()) == 5:
            correct_p = p
            partitions = tmc.get_partitions(p)

    field = np.zeros(Y.shape, dtype=int)
    for k, v in partitions.items():

        x = k // 41
        y = k % 41
        field[v] = 0
        if y < x - 22:
            field[v] = 4
        elif y < 17 - x:
            field[v] = 3
        elif y > x + 23:
            field[v] = 2
        elif y > 63 - x:
            field[v] = 1
        else:
            field[v] = 0

    return field.reshape(grid.shape), correct_p

def create_assignment_map(ensemble, n_clusters, persistence):
    max_points = list()
    max_member = list()

    for i in range(ensemble.shape[2]):
        graph = ngl.EmptyRegionGraph(index=None, max_neighbors=10, relaxed=False, beta=1, p=2.)
        tmc = topopy.MorseComplex(graph=graph,
                                gradient='steepest',
                                normalization='feature')

        X, Y = massage_data(ensemble[:, :, i])
        tmc.build(X, Y)

        for key in tmc.get_partitions(persistence).keys():
            max_points.append((int(X[key, 0]), int(X[key, 1])))
            max_member.append(i)

    maxima = np.array(max_points)
    maxima = MinMaxScaler().fit_transform(maxima)
    clustering = sklearn.cluster.MeanShift().fit(maxima)
    clustering = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters).fit(maxima)
    # clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters).fit(maxima)
    # clustering = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(maxima)
    # clustering = sklearn.cluster.DBSCAN(eps=0.3, min_samples=3).fit(maxima)
    # clustering = sklearn.cluster.SpectralClustering(n_clusters=n_clusters).fit(maxima)
    unique_labels = np.unique(clustering.labels_)
    maxima_map = {}
    for i in range(len(maxima)):
        maxima_map[max_points[i]] = clustering.labels_[i]

    return maxima_map

def assign_labels(grid, maxima_map, persistence):
    X, Y = massage_data(grid)
    h, w = grid.shape

    graph = ngl.EmptyRegionGraph(index=None, max_neighbors=10, relaxed=False, beta=1, p=2.)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization='feature')
    tmc.build(X, Y)

    partitions = tmc.get_partitions(persistence)

    field = np.zeros(Y.shape, dtype=int)
    for k, v in partitions.items():
        field[v] = maxima_map[(int(X[k, 0]), int(X[k, 1]))]


    return field.reshape(grid.shape), persistence

def generate_ensemble(foo, noise_level, count=50, uniform=False):
    xi = np.arange(-1, 1, 0.05)
    xv, yv = np.meshgrid(xi, xi)
    X = np.vstack((xv.flatten(), yv.flatten())).T
    Z = foo(X)
    zv = Z.reshape(xv.shape)
    ground_truth = zv
    uncertain_realizations = np.zeros(shape=ground_truth.shape+(count,))
    np.random.seed(0)
    for i in range(count):
        if uniform:
            uncertain_realizations[:,:, i] = add_uniform_noise(ground_truth, noise_level)
        else:
            uncertain_realizations[:,:, i] = add_nonuniform_noise(ground_truth, noise_level)
    return ground_truth, uncertain_realizations