from matplotlib import colors, cm
import matplotlib.pyplot as plt
from skimage import filters, morphology
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from itertools import cycle

import nglpy
import topopy

from utpy.utils import *

color_list = [[141,211,199], [255,255,179], [190,186,218],
            [251,128,114], [128,177,211], [253,180,98],
            [179,222,105], [252,205,229], [217,217,217]]
# color_list = [[105,239,123], [149,56,144], [192,222,164],
#                 [14,80,62], [153,222,249], [24,81,155],
#                 [218,185,255], [66,30,200], [183,211,33]]
# color_list = [[251,180,174],[179,205,227],[204,235,197],
#               [222,203,228],[254,217,166],[255,255,204],
#               [229,216,189],[253,218,236],[242,242,242]]

ccycle = cycle(color_list)


def show_image(image):
    plt.figure()
    img = plt.imshow(image, cmap=plt.cm.Greys, norm=colors.LogNorm(vmin=image.min(), vmax=image.max()),)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.colorbar(img)

def plot_realization(grid):
    X, Y = massage_data(grid)
    h, w = grid.shape

    graph = nglpy.Graph(index=None, max_neighbors=10, relaxed=False, beta=1, p=2.)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization='feature')
    tmc.build(X, Y)

    partitions = tmc.get_partitions(0.1)
    keys = partitions.keys()

    keyMap = {}
    for i,k in enumerate(keys):
        keyMap[k] = i

    colorList = [
        "#1f78b4",
        "#33a02c",
        "#e31a1c",
        "#ff7f00",
        "#6a3d9a",
        "#b15928",
        "#a6cee3",
        "#b2df8a",
        "#fb9a99",
        "#fdbf6f",
        "#cab2d6",
        "#ffff99",
        "#cccccc",
    ]

    ccycle = cycle(colorList)

    uniqueCount = len(keys)
    usedColors = []
    for i,c in zip(range(uniqueCount), ccycle):
        usedColors.append(c)
    cmap = colors.ListedColormap(usedColors)
    bounds = np.array([keyMap[k] for k in keys]) - 0.5
    bounds = bounds.tolist()
    bounds.append(bounds[-1]+1)
    plt.figure()

    color_mesh = np.zeros((w, h))
    for key, indices in partitions.items():
        for idx in indices:
            color_mesh[idx // w, idx % w] = keyMap[key]

    img = plt.imshow(color_mesh, cmap=cmap, interpolation="nearest", origin="lower")
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.colorbar(img, cmap=cmap, ticks=[range(uniqueCount)], boundaries=bounds)
    plt.contour(grid, cmap=cm.Greys)

def show_persistence_charts(ensemble, my_dir, screen=False):
    plt.figure()

    all_ps = []
    all_counts = []
    for i in range(ensemble.shape[2]):
        graph = nglpy.Graph(index=None, max_neighbors=10, relaxed=False, beta=1, p=2.)
        tmc = topopy.MorseComplex(graph=graph,
                                gradient='steepest',
                                normalization='feature')

        X, Y = massage_data(ensemble[:, :, i])
        tmc.build(X, Y)
        ps = [0]

        count = len(np.unique(list(tmc.get_partitions(0).keys())))
        counts = [count]
        eps = 1e-6
        for i, p in enumerate(tmc.persistences):
            ps.append(p)
            counts.append(count)
            count = len(np.unique(list(tmc.get_partitions(p+eps).keys())))
            ps.append(p)
            counts.append(count)

        all_ps.append(ps)
        all_counts.append(counts)
        plt.plot(ps, counts, alpha=0.2, c='#1f78b4')

    ax = plt.gca()
    ax.set_ylim(0, 25)
    # plt.axhline(2, 0.10, 0.20, linestyle='dashed', color='#000000')
    if screen:
        plt.show()
    plt.savefig("{}/composite_persistence_charts.png".format(my_dir), bbox_inches='tight')
    plt.close()

def show_survival_count(ensemble, my_dir, screen=False):
    all_counts = np.zeros(ensemble[:,:,0].shape)
    all_weighted_counts = np.zeros(ensemble[:,:,0].shape)
    for i in range(ensemble.shape[2]):
    # for i in range(1):
        counts, weighted_counts = count_persistence(ensemble[:,:,i])
        all_counts += counts
        all_weighted_counts += weighted_counts

    plt.figure()
    img = plt.imshow(all_counts, cmap=cm.Greys)
    plt.colorbar(img)
    k = (np.max(all_counts) + np.min(all_counts)) / 2.
    print("Contour visualized for survival count: {}".format(k))
    plt.contour(all_counts, levels=[k], colors='#FFFF00')
    plt.savefig("{}/survival_count.png".format(my_dir), bbox_inches='tight')
    if screen:
        plt.show()
    plt.close()
    return all_counts

def show_weighted_survival_count(ensemble, my_dir, screen=False):
    all_counts = np.zeros(ensemble[:,:,0].shape)
    all_weighted_counts = np.zeros(ensemble[:,:,0].shape)
    for i in range(ensemble.shape[2]):
    # for i in range(1):
        counts, weighted_counts = count_persistence(ensemble[:,:,i])
        all_counts += counts
        all_weighted_counts += weighted_counts

    plt.figure()
    img2 = plt.imshow(all_weighted_counts, cmap=cm.Greys)
    plt.colorbar(img2)
    k = (np.max(all_weighted_counts) + np.min(all_weighted_counts)) / 2.
    print("Contour visualized for weighted survival count: {}".format(k))
    plt.contour(all_weighted_counts, levels=[k], colors='#FFFF00')
    plt.savefig("{}/weighted_survival_count.png".format(my_dir), bbox_inches='tight')
    plt.show()
    plt.close()
    return all_weighted_counts

def show_variance(counts, my_dir, screen=False):
    mean_images = []
    max_radius = 5

    image = (counts - np.min(counts)) / (np.max(counts) - np.min(counts))
    eps=1e-16
    for i in range(1, max_radius):
        mean_images.append(filters.rank.mean(image, selem=morphology.disk(i)))
        if screen:
            show_image(mean_images[-1]+eps)
            plt.gca().set_title("Mean r={}".format(i))
            plt.show()
            plt.close()

    image = 255*image
    variance_images = []
    eps=1e-16
    for i, mean_image in enumerate(mean_images):
        variance_images.append(np.power(image-mean_image,2)+eps)
        show_image(variance_images[-1])
        plt.gca().set_title("Variance r={}".format(i))
        if screen:
            plt.show()
        plt.savefig("{}/weighted_surival_count_variance_{}.png".format(my_dir, i), bbox_inches='tight')
        plt.close()

def show_probabilities_colormap(ensemble, assignments, my_dir, screen=False):
    ps = []
    fields = []
    for i in range(ensemble.shape[2]):
        field, p = assignments(ensemble[:,:,i])
        ps.append(p)
        fields.append(field)

    ps = np.array(ps)
    fields = np.array(fields)
    print(np.min(ps), np.max(ps), np.mean(ps), np.std(ps))

    num_partitions = len(np.unique(fields[0]))
    shape = (num_partitions,) + fields[0].shape
    label_images = np.zeros(shape)

    dim1 = int(np.ceil(np.sqrt(num_partitions)))
    dim2 = int(np.floor(num_partitions/dim1+0.5))
    fig, axes = plt.subplots(dim1, dim2, tight_layout=True)

    axes = axes.flatten()

    for i in range(num_partitions):
        test_image = (fields == i)
        label_images[i] = np.sum(test_image, axis=0)
        img = axes[i].imshow(label_images[i])
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
    for i in range(num_partitions, dim1*dim2):
        axes[i].set_visible(False)

    plt.savefig("{}/partition_probabilities.png".format(my_dir), bbox_inches='tight')
    if screen:
        plt.show()
    plt.close()

def show_blended_overlay(ensemble, assignments, my_dir, screen=False):
    ps = []
    fields = []
    for i in range(ensemble.shape[2]):
        field, p = assignments(ensemble[:,:,i])
        ps.append(p)
        fields.append(field)

    ps = np.array(ps)
    fields = np.array(fields)

    num_partitions = len(np.unique(fields[0]))
    shape = (num_partitions,) + fields[0].shape
    label_images = np.zeros(shape)

    for i in range(num_partitions):
        test_image = (fields == i)
        label_images[i] = np.sum(test_image, axis=0)

    colored_images = []
    for i, c in zip(range(num_partitions), ccycle):
        colored_image = np.zeros(label_images[0].shape + (4,))
        colored_image[:,:,0] = c[0]/255.
        colored_image[:,:,1] = c[1]/255.
        colored_image[:,:,2] = c[2]/255.
        colored_images.append(colored_image)

    for i, label_image in enumerate(label_images):
        colored_images[i][:,:, 3] = label_image / 50.

    plt.figure()
    composite_image = 255*np.ones(colored_images[0].shape)[:,:,:-1]
    for colored_image in colored_images:
        composite_image = overlay_alpha_image_precise(composite_image, 255*colored_image, 1.)
    plt.imshow(composite_image)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().set_ylim(40, 0)
    if screen:
        plt.show()
    plt.savefig("{}/uncertain_assignment_blended_overlay.png".format(my_dir), bbox_inches='tight')
    plt.close()

def show_contour_overlay(ensemble, assignments, my_dir, colored=False, screen=False):
    ps = []
    fields = []
    for i in range(ensemble.shape[2]):
        field, p = assignments(ensemble[:,:,i])
        ps.append(p)
        fields.append(field)

    ps = np.array(ps)
    fields = np.array(fields)

    num_partitions = len(np.unique(fields[0]))
    shape = (num_partitions,) + fields[0].shape
    label_images = np.zeros(shape)

    for i in range(num_partitions):
        test_image = (fields == i)
        label_images[i] = np.sum(test_image, axis=0)

    colored_images = []
    for i, c in zip(range(num_partitions), ccycle):
        colored_image = np.zeros(label_images[0].shape + (4,))
        colored_image[:,:,0] = c[0]/255.
        colored_image[:,:,1] = c[1]/255.
        colored_image[:,:,2] = c[2]/255.
        colored_images.append(colored_image)

    for i, label_image in enumerate(label_images):
        colored_images[i][:,:, 3] = label_image / 50.

    plt.figure()
    for i, color in zip(range(num_partitions), ccycle):
        my_color = "#{}{}{}".format(*[hex(c).split('x')[-1] for c in color])
        if colored:
            plt.contourf(colored_images[i][:,:, 3], levels=[1e-6, 1], colors=my_color, alpha=0.5)
        else:
            plt.contourf(colored_images[i][:,:, 3], levels=[0.99999, 1], colors=my_color, alpha=0.5)
        plt.contour(colored_images[i][:,:, 3], levels=[0.0, 0.5, 1], colors=my_color, linewidths=[1, 0.5, 1.0], linestyles=['solid','dashed','solid'])
    plt.gca().set_ylim(40, 0)
    if screen:
        plt.show()
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig("{}/uncertain_assignment_contour_overlay.png".format(my_dir), bbox_inches='tight')
    plt.close()

def show_combined_overlay(ensemble, assignments, my_dir, contours=False, screen=False):
    ps = []
    fields = []
    for i in range(ensemble.shape[2]):
        field, p = assignments(ensemble[:,:,i])
        ps.append(p)
        fields.append(field)

    ps = np.array(ps)
    fields = np.array(fields)

    num_partitions = len(np.unique(fields[0]))
    shape = (num_partitions,) + fields[0].shape
    label_images = np.zeros(shape)

    for i in range(num_partitions):
        test_image = (fields == i)
        label_images[i] = np.sum(test_image, axis=0)

    colored_images = []
    for i, c in zip(range(num_partitions), ccycle):
        colored_image = np.zeros(label_images[0].shape + (4,))
        colored_image[:,:,0] = c[0]/255.
        colored_image[:,:,1] = c[1]/255.
        colored_image[:,:,2] = c[2]/255.
        colored_images.append(colored_image)

    for i, label_image in enumerate(label_images):
        colored_images[i][:,:, 3] = label_image / 50.

    plt.figure()
    composite_image = 255*np.ones(colored_images[0].shape)[:,:,:-1]
    for colored_image in colored_images:
        composite_image = overlay_alpha_image_precise(composite_image, 255*colored_image, 1.)
    for i, color in zip(range(num_partitions), ccycle):
        my_color = "#{}{}{}".format(*[hex(c).split('x')[-1] for c in color])
        plt.contourf(colored_images[i][:,:, 3], levels=[0.99999, 1], colors="#FFFFFF", alpha=1)
        if contours:
            plt.contour(colored_images[i][:,:, 3], levels=[0.5, 1], colors=my_color, linewidths=[0.5, 1.0], linestyles=['dashed','solid'])

    plt.imshow(composite_image)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.savefig("{}/uncertain_region_assignments.png".format(my_dir), bbox_inches='tight')
    plt.close()

