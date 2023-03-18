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
        # "est_length": np.arange(24, 24 * 2, 24).tolist(),
        # "C_threshold": np.arange(0.5, 1.5, 0.25).tolist(),
        # "delta": np.arange(6, 8, 0.5).tolist(),
        # "est_length": 24,
        # "C_threshold": 1.25,
        # "delta": 7,
        "C_threshold": 14,
        "delta": 4,
        "est_length": 168,
        # "dma_specific": True,
        "default_flow_sensor": "PUMP_1",
    },
    # "mnf": {
    #     # "gamma": np.arange(0, 0.5, 0.05).tolist(),
    #     # "window": [1, 5, 10],
    #     "gamma": 0.15,
    #     "window": 5,
    # },
    # "dualmethod": {"C_threshold": 6.0, "delta": 0.4, "est_length": 888.0},
}


datasets = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)

benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets=datasets,
    results_dir="./grid-search",
    debug=True,
    # multi_parameters=True,
)
# benchmark.add_docker_methods(
#     [
#         # "ghcr.io/ldimbenchmark/lila:0.1.20",
#         "ghcr.io/ldimbenchmark/mnf:0.1.37",
#         # "ghcr.io/ldimbenchmark/dualmethod:0.1.20",
#     ]
# )
# Works fpr containers before 0.1.41
# benchmark.add_local_methods([DUALMethod()])
benchmark.add_local_methods([LILA()])
# benchmark.add_local_methods([DUALMethod()])

# execute benchmark
benchmark.run_benchmark(
    evaluation_mode="training",
    parallel=False,  # parallel_max_workers=4,
    use_cached=False,
)

benchmark.evaluate(
    current_only=True,
    # resultFilter=lambda results: results[results["F1"].notna()],
    write_results=True,
    # generate_plots=True,
)
