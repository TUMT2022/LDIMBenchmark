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
    # "lila": {
    #     # "est_length": np.arange(0.02, 0.25, 0.02).tolist(),
    #     # "C_threshold": np.arange(0, 10, 0.25).tolist(),
    #     # "delta": np.arange(-2, 10, 0.5).tolist(),
    #     # "resample_frequency": ["1T"],
    #     # Best
    #     # "leakfree_time_start": None,
    #     # "leakfree_time_stop": None,
    #     "resample_frequency": "15s",
    #     "est_length": 0.19999999999999998,
    #     "C_threshold": 5.25,
    #     "delta": -1.0,
    #     "default_flow_sensor": "wNode_1",
    # },
    "mnf": {
        "resample_frequency": ["1s", "2s", "4s", "5s", "10s", "15s"],
        "night_flow_interval": ["1T", "2T", "3T", "4T", "5T"],
        "night_flow_start": ["2023-01-01 01:45:00"],
        "gamma": np.around(np.linspace(0, 2.6, 25), 1).tolist(),
        "window": [1, 2, 3, 4, 5, 8, 10, 12, 14, 15],
    },
    # "dualmethod": {
    #     # "est_length": np.arange(0.02, 0.5, 0.02).tolist(),
    #     # "C_threshold": np.arange(0, 10, 0.2).tolist(),
    #     # "delta": np.arange(-2, 10, 0.5).tolist(),
    #     # "resample_frequency": ["1T"],
    #     # Best
    #     "resample_frequency": "15s",
    #     "est_length": 0.12000000000000001,
    #     "C_threshold": 1.8,
    #     "delta": -2.0,
    # },
}

graz = Dataset("test_data/datasets/graz-ragnitz")
# graz._generate_checksum(graz.path)
print(graz.is_valid())

datasets = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)
datasets = [graz]


benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets=datasets,
    results_dir="./graz-ragnitz-test",
    debug=True,
    multi_parameters=True,
)
# benchmark.add_docker_methods(
#     [
#         "ghcr.io/ldimbenchmark/lila:0.2.0",
#         # "ghcr.io/ldimbenchmark/mnf:1.4.0",
#         "ghcr.io/ldimbenchmark/dualmethod:0.1.0",
#     ]
# )
# benchmark.add_local_methods([LILA()])
# benchmark.add_local_methods([DUALMethod()])
benchmark.add_local_methods([MNF()])

benchmark.run_benchmark(
    evaluation_mode="evaluation",
    parallel=True,
    parallel_max_workers=12,
    use_cached=True,
)


results = benchmark.evaluate(
    current_only=True,
    # resultFilter=lambda results: results[results["F1"].notna()],
    write_results=["db", "png"],
    # generate_plots=True,
    print_results=False,
)


# benchmark.evaluate_run(
#     "lila_0.2.0_graz-ragnitz-3c1b5681b7428b322c00316272585a51_evaluation_ecbedbab1883c9f5c2609afec75d4652"
# )
# benchmark.evaluate_run(results.iloc[0]["_folder"])
