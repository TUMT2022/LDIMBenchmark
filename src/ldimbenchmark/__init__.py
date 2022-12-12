"""ldimbenchmark
Main Module
"""


from .benchmark import *
from .classes import *

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        default="config.yml",
        help="config file with arguments (as from this help)",
        metavar="LOG",
    )
    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument(
        "-l",
        "--logLevel",
        dest="loglevel",
        default="INFO",
        help="setting the loglevel",
        metavar="LOG",
    )
    parser.add_argument(
        "-c",
        "--complexity",
        dest="benchmark_complexity",
        default=None,
        choices=["time", "junctions"],
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="benchmark_mode",
        default=None,
        choices=["online", "offline"],
    )

    parser.add_argument(
        "--datasetsFolder",
        dest="datasetsFolder",
        default="./datasets",
        help="root folder containing the datasets",
    )

    parser.add_argument(
        "--datasets",
        dest="datasets",
        default=None,
        action="extend",
        nargs="+",
        type=str,
    )

    parser.add_argument(
        "--algorithms",
        dest="algorithms",
        default=None,
        action="extend",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--resultsFolder", dest="resultsFolder", default="./benchmark-results"
    )
    parser.add_argument(
        "--cacheDir",
        dest="cacheDir",
        default="./cache",
        help="Directory containing cached files (e.g. datasets for complexity analysis)",
    )

    args = parser.parse_args()


# # Enable loading from yaml
# if os.path.exists(args.config):
#     print(f"Loading config from file '{args.config}'")
#     data = yaml.load(open(args.config), Loader=yaml.FullLoader)
#     arg_dict = args.__dict__
#     opt = vars(args)
#     arg_dict = data
#     opt.update(arg_dict)

# print("arguments: {}".format(str(args)))

# if not args.debug:
#     args.resultsFolder = os.path.join(
#         args.resultsFolder, datetime.now().strftime("%Y_%m_%d_%H_%M"))

# os.makedirs(os.path.join(args.resultsFolder), exist_ok=True)

# numeric_level = getattr(logging, args.loglevel.upper(), None)
# if not isinstance(numeric_level, int):
#     raise ValueError('Invalid log level: %s' % args.loglevel)

# fileLogger = logging.FileHandler(os.path.join(
#     args.resultsFolder, "benchmark.log"), mode='w')
# dateFormatter = logging.Formatter(
#     "[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
# )
# fileLogger.setFormatter(dateFormatter)
# logging.basicConfig(
#     level=numeric_level,
#     handlers=[
#         fileLogger,
#         logging.StreamHandler()
#     ]
# )
# logging.getLogger().setLevel(numeric_level)


# algorithms_dir = "./benchmark/algorithms"
# # algorithms = [os.path.join(algorithms_dir, a) for a in args.algorithms]
# algorithms = args.algorithms
# logging.info(f"Using algorithms: {algorithms}")
# algorithm_imports = {}
# for algorithm in algorithms:
#     algorithm_imports[algorithm] = importlib.import_module(
#         "algorithms." + algorithm[:-3]
#     ).CustomAlgorithm

# # Loading datasets
# datasets = glob(args.datasetsFolder + "/*/")

# # Filter datasets list by list given in arguments
# if args.datasets is not None:
#     datasets = [
#         dataset for dataset in datasets if dataset.split("/")[-2] in args.datasets
#     ]

# logging.info(f"Using datasets: {datasets}")
# # Ensure the dataset paths are folders
# datasets = [os.path.join(path) for path in datasets]


## Analysis:

# parser = ArgumentParser()
# parser.add_argument("-d", "--datasets", dest="datasets",  default=None, action="extend", nargs="+", type=str,
#                     help="datasets to include in analysis")
# parser.add_argument("-f", "--datasetsFolder", dest="datasetsFolder", default="./",
#                     help="folder with the datasets")
# parser.add_argument("-o", "--outFolder", dest="outFolder", default="datasets-analysis",
#                     help="folder to put analysis results")

# if not is_notebook():
#     args = parser.parse_args()
# else:
#     args = parser.parse_args([
#         "--datasetsFolder", "../datasets", "--outFolder", "../datasets-analysis"])
