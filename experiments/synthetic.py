import numpy as np
import pandas as pd
from ldimbenchmark.benchmark import LDIMBenchmark
from ldimbenchmark.classes import LDIMMethodBase
from typing import Dict, List
from ldimbenchmark.evaluation_metrics import f1Score
from ldimbenchmark.methods import LILA, MNF, DUALMethod
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
import itertools
import logging

logLevel = "INFO"

numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(
    level=numeric_level,
    handlers=[logging.StreamHandler(), logging.FileHandler("benchmark.log")],
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logging.getLogger().setLevel(numeric_level)

if __name__ == "__main__":
    param_grid = {
        "lila": {
            "synthetic-days-9": {
                "est_length": [
                    2,
                ],
                "C_threshold": [2],
                "delta": [-4],
                "default_flow_sensor": ["sum"],
                "resample_frequency": ["5T"],
            },
        },
        "mnf": {
            "synthetic-days-9": {
                "gamma": [
                    0.5,
                ],
                "window": [5],
            },
        },
        "dualmethod": {
            "synthetic-days-9": {
                "resample_frequency": ["5T"],
                "est_length": [
                    2,
                ],
                "C_threshold": [
                    2,
                ],
                "delta": [
                    -2,
                ],
            },
        },
    }

    datasets = [
        Dataset("./test_data/datasets/synthetic-days-9"),
    ]

    benchmark = LDIMBenchmark(
        hyperparameters=param_grid,
        datasets=datasets,
        results_dir="./grid-search",
        debug=True,
        multi_parameters=True,
    )

    benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/mnf:1.2.0"])
    benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/lila:0.2.0"])
    benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/dualmethod:0.1.0"])

    # # execute benchmark
    benchmark.run_benchmark(
        "evaluation",
        parallel=True,
        parallel_max_workers=4,
        memory_limit="14g",
    )

    benchmark.evaluate(
        write_results=["csv", "tex"],
        current_only=True,
        print_results=True,
        resultFilter=lambda results: results[results["F1"].notna()],
    )

    # benchmark.evaluate_run(
    #     "lila_0.2.0_graz-ragnitz-synthetic-days-90-1324a8dcfa220ea78af25029f094849c"
    # )
