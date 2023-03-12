# %%
# %load_ext autoreload
# %autoreload 2
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
    Dataset(os.path.join(test_data_folder_datasets, "battledim")),
    # Dataset(os.path.join(test_data_folder_datasets, "gjovik")),
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
        60 * 10,
        60 * 20,
        60 * 30,
        60 * 60,
        60 * 60 * 2,
        60 * 60 * 4,
        60 * 60 * 8,
        60 * 60 * 12,
    ],
)
allDerivedDatasets = allDerivedDatasets + derivedDatasets


derivedDatasets = derivator.derive_data("pressures", "sensitivity", [0.1, 0.5, 1, 2, 3])
allDerivedDatasets = allDerivedDatasets + derivedDatasets

derivedDatasets = derivator.derive_model(
    "junctions", "elevation", "accuracy", [16, 8, 4, 2, 1, 0.5, 0.1]
)
allDerivedDatasets = allDerivedDatasets + derivedDatasets

# %%


hyperparameters = {
    "lila": {
        # Best Performances
        "graz-ragnitz": {
            # Overall
            # "resample_frequency": "1T",
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
            # "C_threshold": 14,
            # "est_length": 168,
            # "delta": 4,
        },
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
        },
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

benchmark = LDIMBenchmark(
    hyperparameters,
    allDerivedDatasets,
    # derivedDatasets[0],
    # dataset,
    results_dir="./sensitivity-analysis",
    debug=True,
)
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/lila:0.1.42"])
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/dualmethod:0.1.42"])
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/mnf:0.1.42"])

# benchmark.add_local_methods([MNF()])
# benchmark.add_local_methods([LILA()])

benchmark.run_benchmark(
    evaluation_mode="evaluation",
    parallel=True,
    parallel_max_workers=3,
    # use_cached=False,
)

benchmark.evaluate(
    True,
    write_results="db",
)

# %%

# benchmark.evaluate(True, write_results=True, generate_plots=True)
