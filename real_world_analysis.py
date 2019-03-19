# import matplotlib
# matplotlib.use("Agg")

from utpy.pipeline import analyze_external
import argparse

import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 400

parser = argparse.ArgumentParser(description='Run analysis on a loaded data file')
parser.add_argument('-f', dest='filename', type=str, required=True,
                    help='The filename to load')
parser.add_argument('-n', dest='negate', action="store_true",
                    help='Whether the input data should be negated')
parser.add_argument('-e', dest='n_clusters', type=int, required=False,
                    help='The number of expected extrema')
args = parser.parse_args()

analyze_external(args.filename, negate=args.negate, n_clusters=args.n_clusters)