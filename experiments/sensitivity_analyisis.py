# %%
# %load_ext autoreload
# %autoreload 2
# Fix https://github.com/numpy/numpy/issues/5752
if __name__ == "__main__":
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

    # %%

    # Download
    battledim_dataset = DatasetLibrary(test_data_folder_datasets).download(
        DATASETS.BATTLEDIM
    )[0]

    # %%

    generated_dataset_path = os.path.join(test_data_folder_datasets, "synthetic-days-9")
    generateDatasetForTimeSpanDays(90, generated_dataset_path)
    generated_dataset = Dataset(generated_dataset_path)

    # %%

    dataset = battledim_dataset

    allDerivedDatasets = []
    allDerivedDatasets.append(dataset)

    # %%

    derivator = DatasetDerivator([dataset], os.path.join(test_data_folder_datasets))
    derivedDatasets = derivator.derive_data(
        "pressures", "precision", [0.01, 0.1, 0.5, 1.0]
    )
    allDerivedDatasets = allDerivedDatasets + derivedDatasets

    # #%%

    # derivator = DatasetDerivator([dataset], os.path.join(test_data_folder_datasets))
    # derivedDatasets = derivator.derive_data("pressures", "downsample", [60*10, 60*20, 60*30])
    # allDerivedDatasets = allDerivedDatasets + derivedDatasets

    # #%%

    # derivator = DatasetDerivator([dataset], os.path.join(test_data_folder_datasets))
    # derivedDatasets = derivator.derive_data("pressures", "sensitivity", [1, 0.5, 0.1])
    # allDerivedDatasets = allDerivedDatasets + derivedDatasets

    # %%

    logLevel = "INFO"

    numeric_level = getattr(logging, logLevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % logLevel)

    logging.basicConfig(level=numeric_level, handlers=[logging.StreamHandler()])
    logging.getLogger().setLevel(numeric_level)

    local_methods = [LILA()]

    hyperparameters = {
        "lila": {
            "est_length": 24,
            "C_threshold": 10000.0,
            "delta": 2.0,
        }
    }

    benchmark = LDIMBenchmark(
        hyperparameters,
        allDerivedDatasets,
        # derivedDatasets[0],
        # dataset,
        results_dir="./benchmark-results",
        debug=True,
    )
    benchmark.add_local_methods(local_methods)

    benchmark.run_benchmark(parallel=True, parallel_max_workers=4)

    benchmark.evaluate(True)

    # %%

    benchmark.evaluate(True)

    # benchmark.evaluate(True, write_results=True, generate_plots=True)
