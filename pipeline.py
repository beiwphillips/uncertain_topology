from functools import partial

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

import nglpy
import topopy
import pdir
import os

from utpy.utils import *
from utpy.vis import *
from utpy.test_functions import *

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

def analyze_synthetic(foo="ackley", noise_level=0.3, matlab=False):
    if type(foo) is str:
        matlab = True

    if matlab:
        ground_truth, ensemble, assignments = load_data(foo, noise_level)
    else:
        ground_truth, ensemble = generate_ensemble(foo, noise_level)
        assignments = None

    mean_realization = np.mean(ensemble, axis=2)
    my_dir = "output/{}_{}".format(foo, noise_level)
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)

    plot_realization(ground_truth)
    plt.savefig("{}/gt.png".format(my_dir), bbox_inches='tight')
    plt.close()
    # for i in range(uncertain_realizations.shape[2]):
    #     plot_realization(uncertain_realizations[:,:,i])
    #     plt.savefig("{}/realization_{}.png".format(my_dir, i), bbox_inches='tight')
    #     plt.close()
    plot_realization(mean_realization)
    plt.savefig("{}/realization_mean.png".format(my_dir), bbox_inches='tight')
    plt.close()

    show_persistence_charts(ensemble, my_dir)
    survival_count = show_survival_count(ensemble, my_dir)
    weighted_survival_count = show_weighted_survival_count(ensemble, my_dir)
    show_variance(survival_count, my_dir, True)
    show_variance(weighted_survival_count, my_dir)

    if assignments is None:
        maxima_map = create_assignment_map(ensemble, 9)
        assignments = partial(assign_labels, maxima_map=maxima_map)

    show_probabilities_colormap(ensemble, assignments, my_dir)
    show_blended_overlay(ensemble, assignments, my_dir)
    show_contour_overlay(ensemble, assignments, my_dir)
    show_combined_overlay(ensemble, assignments, my_dir)

analyze_synthetic()