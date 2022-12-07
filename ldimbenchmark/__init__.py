"""
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
