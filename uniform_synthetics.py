import matplotlib
matplotlib.use("Agg")

from utpy.pipeline import analyze_synthetic
import flatpy
import argparse

parser = argparse.ArgumentParser(
    description='Run the entire set of synthetic functions with the specified parameters')
# parser.add_argument('-l', dest='noise_level', type=float, required=True,
#                     help='The level of the noise')
parser.add_argument('-m', dest='noise_model', type=str, default="uniform",
                    help='The type of noise to use')
parser.add_argument('-c', dest='count', type=int, default=50,
                    help='The number of samples to use')
parser.add_argument('-f', dest='functions', type=str, default="",
                    help='A comma-separated list of functions to try')
args = parser.parse_args()

if len(args.functions):
    args.functions = args.functions.strip().split(',')
else:
    args.functions = ["ackley", "himmelblau", "gerber", "gerber_bumpy",
                      "gerber_rotated", "gerber_smeared", "ridge",
                      "ridge_rounded", "checkerBoard", "flatTop",
                      "goldstein_price", "rosenbrock", "salomon",
                      "schwefel", "shekel"]

for noise_level in [1.0]:
    for name, foo in flatpy.twoD.available_functions.items():
        negate = name in ["ackley", "himmelblau"]
        if not len(args.functions) or name in args.functions:
            name = f"{name}_{noise_level}_{args.noise_model}"
            analyze_synthetic(foo, name, noise_level,
                              args.count, args.noise_model, negate)

    for name, foo in flatpy.nD.available_functions.items():
        negate = name in ["ackley", "himmelblau"]
        if not len(args.functions) or name in args.functions:
            name = f"{name}_{noise_level}_{args.noise_model}"
            analyze_synthetic(foo, name, noise_level,
                              args.count, args.noise_model, negate)
