from functools import partial

import os
import numpy as np
import scipy

import utpy.utils as uu
import utpy.vis as uv


def analyze(
    name, ensemble, ground_truth=None, negate=False, n_clusters=None, persistence=None
):
    """ Catch-all function for performing a full suite of analyses on a dataset
        and generating a directory full of images.

    Args:
        name (str): The name of the dataset (dictates where the output folder
            name.
        ensemble (numpy.ndarray): A 3D array where the first two dimensions are
            the data dimensions and
            the third is reserved for the individual realizations.
        ground_truth (numpy.ndarray): A 2D array specifying the ground truth
            function values of a grid of data. Note, passing None means the
            ground truth is unavailable.
        negate (bool): Flag for specifying whether the input should be inverted.
        n_clusters (int): The number of clusters/Morse cells you expect in your
            data.
        persistence (float): The level of simplification. If known, you can set
            this to slightly smaller than the smallest true feature of your
            data.

    Returns:
        None
    """
    if negate:
        ensemble = -ensemble
        if ground_truth is not None:
            ground_truth = -ground_truth

    my_dir = "output/{}".format(name)
    if not os.path.exists(my_dir):
        os.makedirs(my_dir)

    if ground_truth is not None:
        scipy.io.savemat(
            my_dir + "/ensemble.mat", {"ensemble": ensemble, "gt": ground_truth}
        )

    all_ps, all_counts = uv.show_persistence_charts(ensemble, my_dir)
    if persistence is None and n_clusters is None:
        persistence, n_clusters = uu.autotune_from_persistence(all_ps, all_counts)
    elif n_clusters is not None:
        persistence = uu.get_persistence_from_count(ensemble, n_clusters)
    else:
        n_clusters = uu.get_count_from_persistence(ensemble, persistence)

    if ground_truth is not None:
        uv.show_msc(
            ground_truth,
            my_dir,
            persistence=None,
            n_clusters=n_clusters,
            screen=False,
            filename="gt_msc.png",
        )
        uv.show_colormapped_image(ground_truth, my_dir, False, "gt_height.png")
    for i in range(ensemble.shape[2]):
        uv.show_msc(
            ensemble[:, :, i],
            my_dir,
            persistence=None,
            n_clusters=n_clusters,
            screen=False,
            filename="realization_msc_{}.png".format(i),
        )
        uv.show_colormapped_image(
            ensemble[:, :, i], my_dir, False, "realization_height_{}.png".format(i)
        )

    mean_realization = np.mean(ensemble, axis=2)
    uv.show_msc(
        mean_realization,
        my_dir,
        persistence=None,
        n_clusters=n_clusters,
        screen=False,
        filename="realization_mean_msc.png",
    )
    uv.show_colormapped_image(
        mean_realization, my_dir, False, "realization_mean_height.png"
    )
    uv.show_persistence_chart(
        mean_realization, my_dir, False, "mean_persistence_chart.png"
    )

    median_realization = np.median(ensemble, axis=2)
    uv.show_msc(
        median_realization,
        my_dir,
        persistence=None,
        n_clusters=n_clusters,
        screen=False,
        filename="realization_median_msc.png",
    )
    uv.show_colormapped_image(
        median_realization, my_dir, False, "realization_median_height.png"
    )
    uv.show_persistence_chart(
        median_realization, my_dir, False, "median_persistence_chart.png"
    )

    survival_count = uv.show_survival_count(ensemble, my_dir)
    weighted_survival_count = uv.show_weighted_survival_count(ensemble, my_dir)
    weighted_instability_count = uv.show_weighted_instability_count(ensemble, my_dir)
    weighted_consumption_count = uv.show_weighted_consumption_count(ensemble, my_dir)
    max_consumptions = uv.show_max_consumption(ensemble, my_dir)
    uv.show_median_counts(ensemble, my_dir)
    n_clusters_wsc = uu.autotune_from_survival_count(weighted_survival_count)
    uv.show_variance(survival_count, my_dir, True)
    uv.show_variance(weighted_consumption_count, my_dir)

    maxima_map = uu.create_assignment_map(
        ensemble, n_clusters=n_clusters, persistence=persistence
    )
    assignments = partial(
        uu.assign_labels,
        maxima_map=maxima_map,
        n_clusters=n_clusters,
        persistence=persistence,
    )

    uv.show_probabilities_colormap(ensemble, assignments, my_dir)
    uv.show_blended_overlay(ensemble, assignments, my_dir, 0.2)

    uv.show_certain_regions(ensemble, assignments, my_dir, False)
    uv.show_contour_overlay(ensemble, assignments, my_dir, False)
    uv.show_combined_overlay(ensemble, assignments, my_dir, 0.2, True)
    uv.show_combined_overlay(
        ensemble,
        assignments,
        my_dir,
        0.2,
        False,
        filename="uncertain_region_assignments2.png",
    )


def analyze_external(filename, n_clusters=None, negate=False):
    """ Performs a full suite of analyses on a dataset loaded from a csv file

    Args:
        filename (str): The name of the file where the data will be loaded from
            (dictates where the output folder name.
        negate (bool): Flag for specifying whether the input should be inverted.
        n_clusters (int): The number of clusters/Morse cells you expect in your
            data.
        negate (bool): Flag for specifying whether the input should be inverted.

    Returns:
        None
    """
    name = filename
    ensemble = uu.load_ensemble(filename)
    analyze(name, ensemble, negate=negate, n_clusters=n_clusters)


def analyze_synthetic(
    foo, name=None, noise_level=0.3, count=50, noise_model="uniform", negate=False
):
    """ Performs a full suite of analyses on a dataset defined by a closed-form
        function.

    Args:
        foo (function): A function that accepts a 2D numpy.ndarray and returns
            a scalar value for every row of the array.
        name (str): The name of the dataset (dictates where the output folder
            name.
        noise_level (float): A value specifying the maximum amount of noise to
            add to the function in each realization.
        count (int): The number of realizations to generate.
        noise_model (str): One of {'variable', 'uniform', 'nonparametric'}

    Returns:
        None
    """
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

    noise_level = 0.5 * persistence * noise_level
    ground_truth, ensemble = uu.generate_ensemble(foo, noise_level, count, noise_model)
    analyze(
        name,
        ensemble,
        ground_truth,
        negate,
        n_clusters=n_clusters,
        persistence=persistence,
    )
