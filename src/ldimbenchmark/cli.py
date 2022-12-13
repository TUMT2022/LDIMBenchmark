import os
import click
from ldimbenchmark.constants import LDIM_BENCHMARK_CACHE_DIR


class Repo(object):
    def __init__(self, home=None, debug=False):
        self.home = os.path.abspath(home or ".")
        self.debug = debug


@click.group()
@click.option(
    "--config",
    envvar="LDIMBENCHMARK_CONFIG",
    default="config.yml",
    help="config file with arguments (as from this help)",
)
@click.option(
    "--results-folder",
    default="./results/benchmark",
    help="root folder containing the results",
)
@click.option(
    "--cache-dir",
    default=LDIM_BENCHMARK_CACHE_DIR,
    help="root folder containing the results",
)
@click.option(
    "--dataset-folder", default="./datasets", help="root folder containing the datasets"
)
@click.option(
    "--dataset",
    multiple=True,
    help="Select one or multiple datasets from the dataset folder",
)
@click.option(
    "--methods-folder", default="./methods", help="root folder containing the methods"
)
@click.option("--debug/--no-debug", default=False, envvar="DEBUG")
@click.option("--loglevel", default="INFO", help="setting the loglevel", envvar="LOG")
@click.pass_context
def cli(ctx, config, debug):
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
    pass


@cli.command()
@click.option(
    "--complexity", type=click.Choice(["time", "junctions"], case_sensitive=False)
)
@click.option("--mode", type=click.Choice(["online", "offline"], case_sensitive=False))
def benchmark(src, dest):
    pass


@cli.command()
@click.option(
    "--results-folder",
    default="./results/analysis",
    help="out folder containing the analysis results",
)
@click.option("--mode", type=click.Choice(["online", "offline"], case_sensitive=False))
def analysis(src, dest):
    pass


@cli.command()
@click.argument("src")
@click.argument("dest", required=False)
def generator(src, dest):
    pass
