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
    handlers=[logging.StreamHandler(), logging.FileHandler("gjovik.log")],
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logging.getLogger().setLevel(numeric_level)


if __name__ == "__main__":
    param_grid = {
        "lila": {
            #     "est_length": 24,
            #     "C_threshold": 1.25,
            #     "delta": 7,
            # "dma_specific": True,
            "default_flow_sensor": "sum",
        },
        # "mnf": {
        #     "gamma": 0.15,
        #     "window": 5,
        # },
        "dualmethod": {
            "resample_frequency": "60T",
            "C_threshold": 6.0,
            "delta": 0.4,
            "est_length": 888.0,
        },
    }

    # datasets = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)
    datasets = [Dataset("test_data/datasets/gjovik")]

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
    # benchmark.add_local_methods([LILA()])
    benchmark.add_local_methods([DUALMethod()])

    # execute benchmark
    benchmark.run_benchmark(
        evaluation_mode="evaluation", parallel=False  # , parallel_max_workers=4
    )

    benchmark.evaluate(
        current_only=True,
        # resultFilter=lambda results: results[results["F1"].notna()],
        write_results=True,
        # generate_plots=True,
    )
