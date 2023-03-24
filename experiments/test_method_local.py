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
    handlers=[logging.StreamHandler(), logging.FileHandler("output.log")],
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logging.getLogger().setLevel(numeric_level)


# Multi-parameter grid search
param_grid = {
    "lila": {
        # Wide Search
        "est_length": np.arange(24, 24 * 8, 24).tolist(),
        "C_threshold": np.arange(2.0, 16.0, 1).tolist(),
        "delta": np.arange(4, 14, 1).tolist(),
        "default_flow_sensor": ["sum"],
        # Closer Search
        # "est_length": np.arange(24, 24 * 7, 24).tolist(),
        # "C_threshold": np.arange(1.0, 4.0, 1).tolist(),
        # "delta": np.arange(0, 6, 1).tolist(),
        # "default_flow_sensor": ["sum"],
        # "resample_frequency": ["60T", "30T"],
        # Best
        # "est_length": 169.0,
        # "C_threshold": 8.0,
        # "delta": 8.0,
    },
    "mnf": {
        "gamma": np.arange(-1, 1, 0.2).tolist(),
        "window": np.arange(7, 13, 0.2).tolist(),
    },
    "dualmethod": {
        # "est_length": 480.0, "C_threshold": 0.4, "delta": 0.4
        "est_length": [480],  # np.arange(24, 24 * 40, 48).tolist(),
        "C_threshold": np.arange(0, 16, 1).tolist(),  # np.arange(0, 1, 0.2).tolist() +
        "delta": np.arange(0, 16, 1).tolist(),  # np.arange(0, 1, 0.2).tolist() +
        "resample_frequency": ["60T"],
    },
}

# Normal usage
param_grid = {
    # "lila": {"C_threshold": 14, "delta": 4, "est_length": 168},
    "lila": {
        # "leakfree_time_start": None,
        # "leakfree_time_stop": None,
        # "resample_frequency": "30T",
        # "est_length": 144,
        # "C_threshold": 3.0,
        # "delta": 4,
        # "dma_specific": False,
        # "default_flow_sensor": "sum",
        "resample_frequency": "30T",
        "est_length": 144,
        "C_threshold": 3.0,
        "delta": 4,
        "dma_specific": False,
        "default_flow_sensor": "sum",
    },
    # "mnf": {"gamma": -0.10000000000000003, "window": 5.0},
    "mnf": {"resample_frequency": "5T", "window": 12.000000000000004, "gamma": -1.0},
    # "dualmethod":
    # # {"C_threshold": 5.0, "delta": 0.2, "est_length": 936.0},
    # {"C_threshold": 1.0, "delta": 7.0, "est_length": 24.0},
}

# datasets = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)
datasets = [Dataset("test_data/datasets/gjovik")]

# print(datasets[0].id)
benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets=datasets,
    results_dir="./grid-search",
    debug=True,
    # multi_parameters=True,
)

# benchmark.add_local_methods([DUALMethod()])
benchmark.add_local_methods([MNF()])
# benchmark.add_local_methods([LILA()])
# # benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/lila:0.1.42"])

# # execute benchmark
benchmark.run_benchmark(
    "evaluation",
    parallel=True,
    parallel_max_workers=4,
    use_cached=False,
)

benchmark.evaluate(
    # write_results="db",
    print_results=False,
    current_only=True,
    # resultFilter=lambda results: results[results["F1"].notna()],
)
