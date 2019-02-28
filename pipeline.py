from functools import partial


import matplotlib.pyplot as plt
import numpy as np

import nglpy
import topopy
import pdir
import os

from utpy.utils import *
from utpy.vis import *
from utpy.test_functions import *


def analyze_synthetic(foo="ackley", noise_level=0.3, matlab=False):
    if type(foo) is str:
        matlab = True

    if matlab:
        ground_truth, ensemble, assignments = load_data(foo, noise_level)
    else:
        ground_truth, ensemble = generate_ensemble(foo, noise_level)
        assignments = None

    my_dir = "output/{}_{}".format("ackley", noise_level)
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)

    # plot_realization(ground_truth)
    # plt.savefig("{}/gt.png".format(my_dir), bbox_inches='tight')
    # plt.close()
    for i in range(uncertain_realizations.shape[2]):
        plot_realization(uncertain_realizations[:,:,i])
        plt.savefig("{}/realization_{}.png".format(my_dir, i), bbox_inches='tight')
        plt.close()
    mean_realization = np.mean(ensemble, axis=2)
    plot_realization(mean_realization)
    plt.savefig("{}/realization_mean.png".format(my_dir), bbox_inches='tight')
    plt.close()

    all_ps, all_counts = show_persistence_charts(ensemble, my_dir)
    persistence, n_clusters = autotune(all_ps, all_counts)
    print(persistence, n_clusters)
    survival_count = show_survival_count(ensemble, my_dir)
    weighted_survival_count = show_weighted_survival_count(ensemble, my_dir)
    show_variance(survival_count, my_dir, True)
    show_variance(weighted_survival_count, my_dir)

    if assignments is None:
        maxima_map = create_assignment_map(ensemble, n_clusters=n_clusters, persistence=persistence)
        assignments = partial(assign_labels, maxima_map=maxima_map, persistence=persistence)

    show_probabilities_colormap(ensemble, assignments, my_dir)
    show_blended_overlay(ensemble, assignments, my_dir)
    show_contour_overlay(ensemble, assignments, my_dir, True)
    show_contour_overlay(ensemble, assignments, my_dir, False, True)
    show_combined_overlay(ensemble, assignments, my_dir, False, True)
    show_combined_overlay(ensemble, assignments, my_dir, True)

# analyze_synthetic()
analyze_synthetic(ackley, 0.55)