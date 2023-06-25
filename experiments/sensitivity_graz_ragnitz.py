# %%
%load_ext autoreload
%autoreload 2
# Fix https://github.com/numpy/numpy/issues/5752

import os
from ldimbenchmark.datasets.analysis import DatasetAnalyzer
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark.datasets.derivation import DatasetDerivator
from ldimbenchmark.generator import generateDatasetForTimeSpanDays
from ldimbenchmark.methods import MNF, LILA

from ldimbenchmark.benchmark import LDIMBenchmark
import logging
from matplotlib import pyplot as plt
import numpy as np

test_data_folder = "test_data"
test_data_folder_datasets = os.path.join("test_data", "datasets")

logLevel = "INFO"

numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(
    level=numeric_level,
    handlers=[logging.StreamHandler(), logging.FileHandler("analysis.log")],
)
logging.getLogger().setLevel(numeric_level)


# %%

datasets = [
    Dataset(os.path.join(test_data_folder_datasets, "graz-ragnitz")),
]

allDerivedDatasets = datasets

# %%
derivator = DatasetDerivator(
    datasets,
    os.path.join(test_data_folder_datasets),  # ignore_cache=True
)

derivedDatasets = derivator.derive_data(
    "pressures", "precision", [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
)
allDerivedDatasets = allDerivedDatasets + derivedDatasets

derivedDatasets = derivator.derive_data(
    "pressures",
    "downsample",
    [
        10,
        20,
        30,
        60,
        60 * 2,
        60 * 5,
        60 * 10,
    ],
)
allDerivedDatasets = allDerivedDatasets + derivedDatasets


derivedDatasets = derivator.derive_data("pressures", "sensitivity", [0.1, 0.5, 1, 2, 3])
allDerivedDatasets = allDerivedDatasets + derivedDatasets

# TODO Fix derivation function
# derivedDatasets = derivator.derive_model(
#     "junctions", "elevation", "accuracy", [16, 8, 4, 2, 1, 0.5, 0.1]
# )
allDerivedDatasets = allDerivedDatasets + derivedDatasets

# %%


hyperparameters = {
    "lila": {
        # Best Performances
        "graz-ragnitz": {
            # Overall
            "resample_frequency": "15s",
            "est_length": 0.9,
            "C_threshold": 10,
            "delta": -2,
            "dma_specific": False,
            "default_flow_sensor": "wNode_1"
        }
    },
    # not applicable
    # "mnf": { },
    "dualmethod": {
        "graz-ragnitz": {
            # Best Performance Overall
            "resample_frequency": "1T",
            "est_length": 1.1,
            "C_threshold": 0.7,
            "delta": -0.2,
        },
    },
}

benchmark = LDIMBenchmark(
    hyperparameters,
    allDerivedDatasets,
    # derivedDatasets[0],
    # dataset,
    results_dir="./sensitivity-analysis",
    debug=True,
)
# benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/mnf:1.2.0"])
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/lila:0.2.0"])
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/dualmethod:0.1.0"])

benchmark.run_benchmark(
    evaluation_mode="evaluation",
    parallel=True,
    parallel_max_workers=8,
    # use_cached=False,
)

benchmark.evaluate(
    True,
    write_results="db",
)

# %%
