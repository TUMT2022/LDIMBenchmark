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
        # Best Performances
        "graz-ragnitz": {
            # Overall
            "resample_frequency": "1T",
            "est_length": 0.19999999999999998,
            "C_threshold": 5.25,
            "delta": -1.0,
        },
        "battledim": {
            # Training
            "C_threshold": 15,
            "est_length": 168,
            "delta": 12,
            # "Evaluation"
            "C_threshold": 14,
            "est_length": 168,
            "delta": 4,
        }
        # "est_length": np.arange(24, 24 * 8, 24).tolist(),
        # "C_threshold": np.arange(2, 16, 1).tolist(),
        # "delta": np.arange(4, 14, 1).tolist(),
        # Best
        # "est_length": 169.0,
        # "C_threshold": 8.0,
        # "delta": 8.0,
    },
    "mnf": {
        # "gamma": np.arange(-10, 10, 1).tolist(),
        # "gamma": np.arange(-0.3, 1, 0.05).tolist(),
        # "window": [1, 5, 10, 20],
        "battledim": {
            # Best Performance "Training"
            "gamma": 1,
            "window": 10.0,
            # Best Performance "Evaluation"
            # "window": 5.0,
            # "gamma": -0.10000000000000003,
        }
    },
    "dualmethod": {
        "graz-ragnitz": {
            # Best Performance Overall
            "resample_frequency": "1T",
            "est_length": 0.12000000000000001,
            "C_threshold": 1.8,
            "delta": -2.0,
        },
        "battledim": {
            # Best Performance "Training"
            "est_length": 888.0,
            "C_threshold": 0.8,
            "delta": 3.0,
        },
        # Range
        # "est_length": np.arange(24, 24 * 40, 48).tolist(),
        # "C_threshold": np.arange(0, 1, 0.2).tolist() + np.arange(2, 6, 1).tolist(),
        # "delta": np.arange(0, 1, 0.2).tolist() + np.arange(2, 6, 1).tolist(),
    },
}


datasets = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)


benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets=datasets,
    results_dir="./grid-search",
    debug=True,
    # multi_parameters=True,
)
# benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/lila:0.1.20"])
# benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/dualmethod:0.1.20"])
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/mnf:0.1.20"])

# execute benchmark
benchmark.run_benchmark("training", parallel=True, parallel_max_workers=4)

benchmark.evaluate(
    # write_results="database",
    current_only=True,
    # current_only=False,
    # resultFilter=lambda results: results[results["F1"].notna()],
)
