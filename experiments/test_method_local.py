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
    handlers=[logging.StreamHandler(), logging.FileHandler("output.log")],
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logging.getLogger().setLevel(numeric_level)


# Multi-parameter grid search
param_grid = {
    "lila": {
        "est_length": np.arange(24, 24 * 8, 24).tolist(),
        "C_threshold": np.arange(2.0, 16.0, 1).tolist(),
        "delta": np.arange(4, 14, 1).tolist(),
        # Best
        # "est_length": 169.0,
        # "C_threshold": 8.0,
        # "delta": 8.0,
    },
    "mnf": {
        "gamma": np.arange(-0.3, 1, 0.05).tolist(),
        "window": [1, 5, 10, 20],
    },
    "dualmethod": {
        # "est_length": 480.0, "C_threshold": 0.4, "delta": 0.4
        "est_length": np.arange(24, 24 * 40, 48).tolist(),
        "C_threshold": np.arange(0, 1, 0.2).tolist() + np.arange(2, 6, 1).tolist(),
        "delta": np.arange(0, 1, 0.2).tolist() + np.arange(2, 6, 1).tolist(),
    },
}

# Normal usage
param_grid = {
    "lila": {"C_threshold": 14, "delta": 4, "est_length": 168},
    "mnf": {"gamma": -0.10000000000000003, "window": 5.0},
    "dualmethod": {"C_threshold": 5.0, "delta": 0.2, "est_length": 936.0},
}

datasets = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)


benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets="test_data/datasets/battledim",
    results_dir="./grid-search",
    debug=True,
    # multi_parameters=True,
)
local_methods = [MNF()]  # LILA()
benchmark.add_local_methods(local_methods)
# benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/lila:0.1.20"])

# execute benchmark
benchmark.run_benchmark(
    "training", parallel=True, parallel_max_workers=6, use_cached=False
)

benchmark.evaluate(
    write_results=True,
    current_only=True,
    resultFilter=lambda results: results[results["F1"].notna()],
)
