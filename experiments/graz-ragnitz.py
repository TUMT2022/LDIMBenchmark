import numpy as np
import pandas as pd
from ldimbenchmark.benchmark import LDIMBenchmark
from ldimbenchmark.classes import LDIMMethodBase
from typing import Dict, List
from ldimbenchmark.evaluation_metrics import f1Score
from ldimbenchmark.methods import LILA, MNF
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
import itertools
import logging

from ldimbenchmark.methods.dualmethod import DUALMethod

logLevel = "INFO"

numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(
    level=numeric_level,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logging.getLogger().setLevel(numeric_level)


param_grid = {
    "lila": {
        # "est_length": np.arange(0.02, 0.25, 0.02).tolist(),
        # "C_threshold": np.arange(0, 10, 0.25).tolist(),
        # "delta": np.arange(-2, 10, 0.5).tolist(),
        # "resample_frequency": ["1T"],
        # Best
        # "leakfree_time_start": None,
        # "leakfree_time_stop": None,
        "resample_frequency": "1T",
        "est_length": 0.19999999999999998,
        "C_threshold": 5.25,
        "delta": -1.0,
    },
    "mnf": {
        "gamma": np.arange(0, 0.5, 0.05).tolist(),
        "window": [1, 5, 10],
        # "gamma": 0.15,
        # "window": 5,
    },
    "dualmethod": {
        # "est_length": np.arange(0.02, 0.5, 0.02).tolist(),
        # "C_threshold": np.arange(0, 10, 0.2).tolist(),
        # "delta": np.arange(-2, 10, 0.5).tolist(),
        # "resample_frequency": ["1T"],
        # Best
        "resample_frequency": "1T",
        "est_length": 0.12000000000000001,
        "C_threshold": 1.8,
        "delta": -2.0,
    },
}


# datasets = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)
datasets = [Dataset("test_data/datasets/graz-ragnitz")]


benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets=datasets,
    results_dir="./grid-search",
    # debug=True,
    # multi_parameters=True,
)
# benchmark.add_docker_methods(
#     [
#         "ghcr.io/ldimbenchmark/lila:0.1.39",
#         # "ghcr.io/ldimbenchmark/mnf:0.1.38",
#         "ghcr.io/ldimbenchmark/dualmethod:0.1.39",
#     ]
# )
# benchmark.add_local_methods([LILA()])
# benchmark.add_local_methods([DUALMethod()])

# benchmark.run_benchmark(
#     evaluation_mode="evaluation", parallel=True, parallel_max_workers=10
# )

benchmark.evaluate(
    current_only=True,
    # resultFilter=lambda results: results[results["F1"].notna()],
    write_results="db",
    # generate_plots=True,
)
