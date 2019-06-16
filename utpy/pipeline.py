from functools import partial

import os
import numpy as np
import scipy

from utpy.utils import (autotune_from_survival_count,
                        autotune_from_persistence,
                        assign_labels,
                        create_assignment_map,
                        generate_ensemble,
                        load_ensemble,
                        get_count_from_persistence,
                        get_persistence_from_count)
from utpy.vis import (show_colormapped_image,
                      show_persistence_charts,
                      show_persistence_chart,
                      show_survival_count,
                      show_weighted_survival_count,
                      show_weighted_instability_count,
                      show_weighted_consumption_count,
                      show_max_consumption,
                      show_median_counts,
                      show_variance,
                      show_probabilities_colormap,
                      show_blended_overlay,
                      show_contour_overlay,
                      show_certain_regions,
                      show_combined_overlay,
                      show_msc,)


def analyze(name, ensemble, ground_truth=None, negate=False, n_clusters=None, persistence=None):
    if negate:
        ensemble = -ensemble
        if ground_truth is not None:
            ground_truth = -ground_truth

    my_dir = "output/{}".format(name)
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)

    if ground_truth is not None:
        scipy.io.savemat(my_dir + "/ensemble.mat", {"ensemble": ensemble, "gt": ground_truth})

    all_ps, all_counts = show_persistence_charts(ensemble, my_dir)
    if persistence is None and n_clusters is None:
        persistence, n_clusters = autotune_from_persistence(all_ps, all_counts)
    elif n_clusters is not None:
        persistence = get_persistence_from_count(ensemble, n_clusters)
    else:
        n_clusters = get_count_from_persistence(ensemble, persistence)

    if ground_truth is not None:
        show_msc(ground_truth, my_dir, persistence=None, n_clusters=n_clusters, screen=False, filename="gt_msc.png")
        show_colormapped_image(ground_truth, my_dir, False, "gt_height.png")
    for i in range(ensemble.shape[2]):
        show_msc(ensemble[:, :, i], my_dir, persistence=None, n_clusters=n_clusters, screen=False,
                 filename="realization_msc_{}.png".format(i))
        show_colormapped_image(
            ensemble[:, :, i], my_dir, False, "realization_height_{}.png".format(i))

    mean_realization = np.mean(ensemble, axis=2)
    show_msc(mean_realization, my_dir, persistence=None, n_clusters=n_clusters,
             screen=False, filename="realization_mean_msc.png")
    show_colormapped_image(mean_realization, my_dir,
                           False, "realization_mean_height.png")
    show_persistence_chart(mean_realization, my_dir, False, "mean_persistence_chart.png")

    median_realization = np.median(ensemble, axis=2)
    show_msc(median_realization, my_dir, persistence=None, n_clusters=n_clusters,
             screen=False, filename="realization_median_msc.png")
    show_colormapped_image(median_realization, my_dir,
                           False, "realization_median_height.png")
    show_persistence_chart(median_realization, my_dir, False, "median_persistence_chart.png")

    survival_count = show_survival_count(ensemble, my_dir)
    weighted_survival_count = show_weighted_survival_count(ensemble, my_dir)
    weighted_instability_count = show_weighted_instability_count(ensemble, my_dir)
    weighted_consumption_count = show_weighted_consumption_count(ensemble, my_dir)
    max_consumptions = show_max_consumption(ensemble, my_dir)
    show_median_counts(ensemble, my_dir)
    n_clusters_wsc = autotune_from_survival_count(weighted_survival_count)
    show_variance(survival_count, my_dir, True)
    show_variance(weighted_consumption_count, my_dir)

    maxima_map = create_assignment_map(
        ensemble, n_clusters=n_clusters, persistence=persistence)
    assignments = partial(
        assign_labels, maxima_map=maxima_map, n_clusters=n_clusters, persistence=persistence)

    show_probabilities_colormap(ensemble, assignments, my_dir)
    show_blended_overlay(ensemble, assignments, my_dir, 0.2)

    show_certain_regions(ensemble, assignments, my_dir, False)
    show_contour_overlay(ensemble, assignments, my_dir, False)
    show_combined_overlay(ensemble, assignments, my_dir, 0.2, True)
    show_combined_overlay(ensemble, assignments, my_dir, 0.2, False, filename="uncertain_region_assignments2.png")


def analyze_external(filename, n_clusters=None, negate=False):
    name = filename
    ensemble = load_ensemble(filename)
    analyze(name, ensemble, negate=negate, n_clusters=n_clusters)


def analyze_synthetic(foo,
                      name=None,
                      noise_level=0.3,
                      count=50,
                      noise_model="uniform",
                      negate=False):
    if name is None:
        name = foo.__name__
    if "ackley" in name:
        persistence = 0.665
        n_clusters = 9
    elif "checkerBoard" in name:
        persistence = 1.5
        n_clusters = 2
    elif "diagonal" in name:
        persistence = 0.6
        n_clusters = 3
    elif "flatTop" in name:
        persistence = 1
        n_clusters = 2
    elif "rosenbrock" in name:
        persistence = 1000
        n_clusters = 3
    elif "salomon" in name:
        persistence = 1.9
        n_clusters = 5
    elif "schwefel" in name:
        persistence = 150
        n_clusters = 49
    elif "shekel" in name:
        persistence = 1.5
        n_clusters = 4
    elif "_smeared" in name:
        persistence = 0.8
        n_clusters = 2
    elif "_bumpy" in name:
        persistence = 0.145
        n_clusters = 8
    elif "gerber" in name:
        persistence = 0.25
        n_clusters = 4
    elif "goldstein_price" in name:
        persistence = 140000
        n_clusters = 5
    elif "hill" in name:
        persistence = 1
        n_clusters = 1
    elif "himmelblau" in name:
        persistence = 400
        n_clusters = 4
    elif "hinge" in name:
        persistence = 0.5
        n_clusters = 2
    elif "ridge" in name:
        persistence = 0.5
        n_clusters = 3
    elif "strangulation" in name:
        persistence = 0.9
        n_clusters = 1

    noise_level = 0.5*persistence*noise_level
    ground_truth, ensemble = generate_ensemble(
        foo, noise_level, count, noise_model)
    analyze(name, ensemble, ground_truth, negate, n_clusters=n_clusters, persistence=persistence)
