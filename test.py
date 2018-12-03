from MorseComplex import *
from matplotlib import colors
import matplotlib.pyplot as plt
import scipy.io
import itertools

xComponent = scipy.io.loadmat("data/xComponent")["discreteGradProbX"]
yComponent = scipy.io.loadmat("data/yComponent")["discreteGradProbY"]
mandatoryMaxima = scipy.io.loadmat("data/mandatoryMax")["mandatoryMax"]

mc = MorseComplex2D(xComponent, yComponent, mandatoryMaxima)

flow = mc.maxFlow
# print(np.unique(flow))
flow[np.where(flow < 0)] = -1
uniqueCount = len(np.unique(flow))

colorList = [
    "#cccccc",
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
]

########################################################################

cmap = colors.ListedColormap(colorList[:uniqueCount])
bounds = np.unique(flow) - 0.5
bounds = bounds.tolist()
bounds.append(bounds[-1]+1)
img = plt.imshow(flow, cmap=cmap, interpolation="nearest", origin="lower")
plt.colorbar(img, cmap=cmap, ticks=np.unique(flow), boundaries=bounds)

########################################################################

# colorCycle = itertools.cycle(colorList)
# colorMap = {}

# ids = {}
# for i in np.unique(mc.maxFlow):
#     colorMap[i] = next(colorCycle)
#     ids[i] = []
#     for row, vals in enumerate(mc.maxFlow):
#         for col, val in enumerate(vals):
#             if val == i:
#                 ids[i].append([col, row])

# for key, values in ids.items():
#     positions = np.array(values)
#     plt.scatter(positions[:, 0], positions[:, 1], c=colorMap[key])
plt.show()
