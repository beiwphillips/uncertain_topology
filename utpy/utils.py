from glob import glob
import scipy.io
import sklearn.cluster
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import nglpy_cuda as ngl
import topopy
import flatpy

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

graph_params = {
    "index": None, "max_neighbors": 10, "relaxed": False, "beta": 1, "p": 2.
}


def load_data(foo="ackley", noise_level=0.3):
    assignment_map = {"4peaks": assignments_4peaks,
                      "ackley": assignments_ackley, "salomon": assignments_salomon}
    base_name = "data/" + foo

    ground_truth = scipy.io.loadmat(base_name + "/groundtruth.mat")['gt']
    uncertain_realizations = scipy.io.loadmat(
        "{}/{}_uncertain.mat".format(base_name, str(noise_level)))['noisyEnsemble']

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
            uncertain_realizations.append(
                scipy.io.loadmat(filename)[token][:, :, 0].T)
        else:
            uncertain_realizations.append(scipy.io.loadmat(filename)[token].T)
    return np.array(uncertain_realizations).T


def massage_data(grid):
    X = []
    Y = []
    for row, vals in enumerate(grid):
        for col, val in enumerate(vals):
            X.append([col, row])
            Y.append(val)
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)


def count_persistence(grid):
    X, Y = massage_data(grid)
    h, w = grid.shape

    graph = ngl.EmptyRegionGraph(**graph_params)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization=None)
    tmc.build(X, Y)

    partitions = tmc.get_partitions()
    sorted_hierarchy = sorted([(p, k, x)
                               for k, (p, x, s) in tmc.merge_sequence.items()])

    field = np.zeros(Y.shape, dtype=int)
    for k, v in partitions.items():
        field[v] = k

    counts = np.zeros(Y.shape)
    weighted_counts = np.zeros(Y.shape)
    unstable_counts = np.zeros(Y.shape)
    consumption_counts = np.zeros(Y.shape)
    last_persistence = 0
    for persistence, dying_index, surviving_index in sorted_hierarchy:
        next_field = np.array(field)
        next_field[np.where(next_field == dying_index)] = surviving_index

        counts[np.where(field == next_field)] += 1
        weighted_counts[np.where(field == next_field)
                        ] += (persistence-last_persistence)
        unstable_counts[np.where(field != next_field)
                        ] += (persistence-last_persistence)
        consumption_counts[np.where(
            field == surviving_index)] += (persistence-last_persistence)
        field = next_field
        last_persistence = persistence

    return counts.reshape(grid.shape), weighted_counts.reshape(grid.shape), unstable_counts.reshape(grid.shape), consumption_counts.reshape(grid.shape)


def get_persistence_from_count(ensemble, n_clusters):
    persistences = []
    for i in range(ensemble.shape[2]):
        graph = ngl.EmptyRegionGraph(**graph_params)
        tmc = topopy.MorseComplex(graph=graph,
                                  gradient='steepest',
                                  normalization=None)

        X, Y = massage_data(ensemble[:, :, i])
        tmc.build(X, Y)
        for p in tmc.persistences:
            if len(tmc.get_partitions(p).keys()) <= n_clusters:
                persistences.append(p)
                break
    return np.average(persistences)


def get_count_from_persistence(ensemble, persistence):
    max_counts = []
    for i in range(ensemble.shape[2]):
        graph = ngl.EmptyRegionGraph(**graph_params)
        tmc = topopy.MorseComplex(graph=graph,
                                  gradient='steepest',
                                  normalization=None)
        X, Y = massage_data(ensemble[:, :, i])
        tmc.build(X, Y)
        max_counts.append(len(tmc.get_partitions(persistence).keys()))
    return int(np.average(max_counts))


def create_assignment_map(ensemble, n_clusters=None, persistence=None):
    if n_clusters is None and persistence is None:
        raise ValueError("Must specify either n_clusters or persistence")
    max_points = list()
    max_member = list()

    max_counts = []
    for i in range(ensemble.shape[2]):
        graph = ngl.EmptyRegionGraph(**graph_params)
        tmc = topopy.MorseComplex(graph=graph,
                                  gradient='steepest',
                                  normalization=None)

        X, Y = massage_data(ensemble[:, :, i])
        tmc.build(X, Y)
        if n_clusters is None and persistence is not None:
            max_counts.append(len(tmc.get_partitions(persistence).keys()))
            for key in tmc.get_partitions(persistence).keys():
                max_points.append((int(X[key, 0]), int(X[key, 1])))
                max_member.append(i)
        else:
            for p in tmc.persistences:
                if len(tmc.get_partitions(p).keys()) <= n_clusters:
                    for key in tmc.get_partitions(p).keys():
                        max_points.append((int(X[key, 0]), int(X[key, 1])))
                        max_member.append(i)
                    break
    if n_clusters is None and persistence is not None:
        n_clusters = int(np.average(max_counts))

    maxima = np.array(max_points)
    maxima = MinMaxScaler().fit_transform(maxima)
    clustering = sklearn.cluster.MeanShift().fit(maxima)
    clustering = sklearn.cluster.MiniBatchKMeans(
        n_clusters=n_clusters).fit(maxima)
    # clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters).fit(maxima)
    # clustering = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(maxima)
    # clustering = sklearn.cluster.DBSCAN(eps=0.3, min_samples=3).fit(maxima)
    # clustering = sklearn.cluster.SpectralClustering(n_clusters=n_clusters).fit(maxima)
    unique_labels = np.unique(clustering.labels_)
    maxima_map = {}
    for i in range(len(maxima)):
        maxima_map[max_points[i]] = clustering.labels_[i]

    return maxima_map


def assign_labels(grid, maxima_map, n_clusters=None, persistence=None):
    if n_clusters is None and persistence is None:
        raise ValueError("Must specify either n_clusters or persistence")

    X, Y = massage_data(grid)
    h, w = grid.shape

    graph = ngl.EmptyRegionGraph(**graph_params)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization=None)
    tmc.build(X, Y)

    if n_clusters is None and persistence is not None:
        partitions = tmc.get_partitions(persistence)
    else:
        for p in tmc.persistences:
            if len(tmc.get_partitions(p).keys()) <= n_clusters:
                persistence = p
                partitions = tmc.get_partitions(p)
                break

    field = np.zeros(Y.shape, dtype=int)
    for k, v in partitions.items():
        field[v] = maxima_map[(int(X[k, 0]), int(X[k, 1]))]

    return field.reshape(grid.shape), persistence


def generate_ensemble(foo, noise_level, count=50, noise_model="uniform"):
    xi = np.arange(0, 1, 0.025)
    xv, yv = np.meshgrid(xi, xi)
    X = np.vstack((xv.flatten(), yv.flatten())).T
    Z = foo(X)
    graph = ngl.EmptyRegionGraph(**graph_params)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization=None)
    tmc.build(X, Z)

    partitions = tmc.get_partitions()
    sorted_hierarchy = sorted([(p, len(tmc.get_partitions(p)))
                               for k, (p, x, s) in tmc.merge_sequence.items()])

    # This is handled outside this function at the analyze_synthetic level
    # z_range = max(Z) - min(Z)
    # noise = 0.5*z_range*noise_level
    noise = noise_level
    zv = Z.reshape(xv.shape)
    ground_truth = zv
    uncertain_realizations = np.zeros(shape=ground_truth.shape+(count,))
    np.random.seed(0)
    for i in range(count):
        if noise_model == "uniform":
            uncertain_realizations[:, :, i] = flatpy.utils.add_uniform_noise(
                ground_truth, noise)
        elif noise_model == "nonparametric":
            uncertain_realizations[:, :, i] = flatpy.utils.add_nonparametric_uniform_noise(
                ground_truth, noise, 0.2, 10*noise)
        elif noise_model == "variable":
            uncertain_realizations[:, :, i] = flatpy.utils.add_nonuniform_noise(
                ground_truth, noise)
    return ground_truth, uncertain_realizations


def autotune_from_persistence(all_ps, all_counts):
    unique_persistences = np.array(
        sorted(set([p for ps in all_ps for p in ps])))
    unique_counts = np.zeros(shape=(len(unique_persistences), len(all_ps)))
    for row, p in enumerate(unique_persistences):
        for col in range(len(all_ps)):
            index = 0
            while index < len(all_ps[col])-1 and all_ps[col][index] < p:
                index += 1
            unique_counts[row, col] = all_counts[col][index]

    first_saved = None
    for p, c in zip(unique_persistences, unique_counts):
        mu = np.mean(c)
        sigma = np.std(c)
        counts = np.bincount(np.array(c, dtype=int))
        max_count = np.argmax(counts)
        if first_saved is None and counts[max_count] >= 0.9*len(all_ps):
            first_saved = (p, max_count)
    plt.figure()

    x = unique_persistences
    y = np.mean(unique_counts, axis=1)
    sigma = np.std(unique_counts, axis=1)

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, sigma.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(sigma)
    lc.set_linewidth(4)
    line = plt.gca().add_collection(lc)
    plt.colorbar(line)
    plt.gca().set_xlim(0, np.max(unique_persistences))
    plt.gca().set_ylim(0, np.max(unique_counts))
    plt.show()

    # plt.figure()
    # df = pd.DataFrame({"Member": memberships, "Persistence": unique_persistences, "Maximum_Count": counts})
    # sns.lineplot(x="Persistence", y="Maximum_Count", data=df)
    # plt.show()

    return first_saved


def autotune_from_survival_count(counts):
    # Perform image-based intensity segmentation here
    clustering = sklearn.cluster.DBSCAN(eps=0.3, min_samples=3).fit(counts)
    return
