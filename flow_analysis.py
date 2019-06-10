import utpy.utils
import utpy.vis
import flatpy
from functools import partial
from itertools import cycle

mandatory_maxima = scipy.io.loadmat("data/upSampled/flowMandatoryMaxima.matlab")['mandatoryMax']
filename = "upSampled"
n_clusters = len(np.unique(mandatory_maxima))-1
my_dir = "output/{}".format(filename)
ensemble = utpy.utils.load_ensemble(filename)

def find_closest_index(points, x):
    x = np.array(x)
    return np.argmin(np.linalg.norm(points - x, axis=1))
mandatory_maxima = scipy.io.loadmat("data/upSampled/flowMandatoryMaxima.matlab")['mandatoryMax']
max_points = list()
max_counts = []
for i in range(ensemble.shape[2]):
    graph = ngl.EmptyRegionGraph(**utpy.utils.graph_params)
    tmc = topopy.MorseComplex(graph=graph,
                                gradient='steepest',
                                normalization=None)

    X, Y = utpy.utils.massage_data(ensemble[:, :, i])
    tmc.build(X, Y)
    for p in tmc.persistences:
        if len(tmc.get_partitions(p).keys()) <= n_clusters:
            for key in tmc.get_partitions(p).keys():
                max_points.append((int(X[key, 0]), int(X[key, 1])))
            break

points = []
labels = []
for row in range(mandatory_maxima.shape[0]):
    for col in range(mandatory_maxima.shape[1]):
        if mandatory_maxima[row, col] != -1:
            points.append((col, row))
            labels.append(mandatory_maxima[row, col])
points = np.array(points)

maxima_map = {}
for i in range(len(max_points)):
    maxima_map[max_points[i]] = labels[find_closest_index(points, max_points[i])]

mean_realization = np.mean(ensemble, axis=2)
median_realization = np.median(ensemble, axis=2)

def show_msc(grid, persistence=None, n_clusters=None, color="#000000"):
    X, Y = utpy.utils.massage_data(grid)
    h, w = grid.shape

    graph = ngl.EmptyRegionGraph(**graph_params)
    tmc = topopy.MorseComplex(graph=graph,
                              gradient='steepest',
                              normalization=None)
    tmc.build(X, Y)

    if persistence is None:
        for p in tmc.persistences:
            if len(tmc.get_partitions(p).keys()) <= n_clusters:
                persistence = p
                break

    partitions = tmc.get_partitions(persistence)
    keys = partitions.keys()

    keyMap = {}
    levels = []
    for i, k in enumerate(keys):
        keyMap[k] = i
        levels.append(i+0.5)

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
    for i, c in zip(range(uniqueCount), ccycle):
        usedColors.append(c)
    cmap = colors.ListedColormap(usedColors)
    bounds = np.array([keyMap[k] for k in keys]) - 0.5
    bounds = bounds.tolist()
    bounds.append(bounds[-1]+1)

    color_mesh = np.zeros((h, w))
    for key, indices in partitions.items():
        for idx in indices:
            color_mesh[idx // w, idx % w] = keyMap[key]

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    for i in keyMap.values():
        lines = plt.contour(color_mesh == i, colors=color, levels=levels, linewidths=1)
    plt.gca().set_aspect("equal")
    return lines

def show_contour_overlay(ensemble, assignments):
    ps = []
    fields = []
    count = ensemble.shape[2]
    for i in range(count):
        field, p = assignments(ensemble[:, :, i])
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
        colored_image[:, :, 0] = c[0]/255.
        colored_image[:, :, 1] = c[1]/255.
        colored_image[:, :, 2] = c[2]/255.
        colored_images.append(colored_image)

    for i, label_image in enumerate(label_images):
        colored_images[i][:, :, 3] = label_image / count

    for i, color in zip(range(num_partitions), ccycle):
        my_color = "#{:>02}{:>02}{:>02}".format(*[hex(c).split('x')[-1] for c in color])
        plt.contourf(colored_images[i][:, :, 3], levels=[0.99999, 1], colors=my_color, alpha=0.5)
#         if colored:
#             plt.contourf(colored_images[i][:, :, 3], levels=[
#                          1e-6, 1], colors=my_color, alpha=0.5)
#         else:
#             plt.contourf(colored_images[i][:, :, 3], levels=[
#                          0.99999, 1], colors=my_color, alpha=0.5)
        plt.contour(colored_images[i][:, :, 3], levels=[0.0, 0.5, 1], colors=my_color, linewidths=[
                    1, 0.5, 1.0], linestyles=['solid', 'dashed', 'solid'])
    plt.gca().set_xlim(0, ensemble.shape[1])
    # plt.gca().set_ylim(ensemble.shape[0], 0)
    plt.gca().set_ylim(0, ensemble.shape[0])
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().set_aspect('equal')

persistence = utpy.utils.get_persistence_from_count(ensemble, n_clusters)
# maxima_map = utpy.utils.create_assignment_map(ensemble, n_clusters=n_clusters, persistence=persistence)
assignments = partial(utpy.utils.assign_labels, maxima_map=maxima_map, n_clusters=n_clusters, persistence=persistence)

def show_combined_overlay(ensemble, assignments, gamma=2.2, color="#000000"):
    fig, ax = plt.subplots(3, 1, dpi=200)
    ps = []
    fields = []
    count = ensemble.shape[2]
    for i in range(count):
        field, p = assignments(ensemble[:, :, i])
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
        colored_image[:, :, 0] = c[0]/255.
        colored_image[:, :, 1] = c[1]/255.
        colored_image[:, :, 2] = c[2]/255.
        colored_images.append(colored_image)

    certain_mask = np.zeros(label_images[0].shape, dtype=bool)
    for i, label_image in enumerate(label_images):
        colored_images[i][:, :, 3] = label_image / count
        certain_mask = np.logical_or(certain_mask, label_image / count == 1)

    composite_image = 255*np.ones(colored_images[0].shape)[:, :, :-1]
    for colored_image in colored_images:
        composite_image = utpy.vis.overlay_alpha_image_precise(
            composite_image, 255*colored_image, 1.1, gamma)
    for i, triplet in zip(range(num_partitions), ccycle):
        my_color = "#{:>02}{:>02}{:>02}".format(*[hex(c).split('x')[-1] for c in triplet])
        ax[1].contourf(colored_images[i][:, :, 3], levels=[
                     0.99999, 1], colors="#FFFFFF", alpha=1)
        ax[2].contourf(colored_images[i][:, :, 3], levels=[
                     0.99999, 1], colors="#FFFFFF", alpha=1)
#         ax[1].contourf(colored_images[i][:, :, 3], levels=[0.99999, 1], colors=my_color, alpha=1)
        ax[1].contour(colored_images[i][:, :, 3], levels=[0.5], colors="#000000", linewidths=[1], linestyles=['solid'])
        ax[2].contour(colored_images[i][:, :, 3], levels=[0.5], colors="#000000", linewidths=[1], linestyles=['solid'])


#     certain_image = np.ma.masked_where(np.dstack((certain_mask,)*3), composite_image)
#     ax[0].imshow(certain_image)
    ax[1].imshow(composite_image)
    ax[2].imshow(composite_image)
    for a in ax:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        # a.set_ylim(ensemble.shape[0], 0)
        a.set_ylim(0, ensemble.shape[0])

color_list = []
for i in range(20):
    color_list.append([int(255*i) for i in plt.cm.tab20(i)])

ccycle = cycle(color_list)

umc_color = "#000000"
gt_color = "#4daf4a"
mean_color = "#e41a1c"
show_combined_overlay(ensemble, assignments, 0.2, umc_color)
# show_msc(ground_truth, n_clusters=n_clusters, color=gt_color)
show_msc(mean_realization, n_clusters=n_clusters, color=mean_color)
# plt.plot([-1,-0.5], [0,1], color=gt_color, linewidth=1, label="Truth")
plt.gca().plot([-1,-0.5], [0,1], color=mean_color, linewidth=1, label="Mean")
plt.gca().plot([-1,-0.5], [0,1], color=umc_color, linewidth=1, label="50% Prob.")
plt.gca().set_xlim(0, ensemble.shape[1])
plt.gca().set_ylim(0 ,ensemble.shape[0])
_ = plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=3)
