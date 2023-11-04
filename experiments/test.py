# %%
%load_ext autoreload
%autoreload 2
from ldimbenchmark.datasets.analysis import DatasetAnalyzer
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark.datasets.derivation import DatasetDerivator
from ldimbenchmark.generator import generateDatasetForTimeSpanDays
from ldimbenchmark.methods import MNF, LILA

from ldimbenchmark.benchmark import LDIMBenchmark
import logging
import os
from matplotlib import pyplot as plt

test_data_folder = "test_data"
test_data_folder_datasets = os.path.join("test_data", "datasets")

# %%
datasets = [
    Dataset(os.path.join(test_data_folder_datasets, "graz-ragnitz")),
]
analyzer = DatasetAnalyzer("sensitivity-analysis")
derivator = DatasetDerivator(
    datasets,
    os.path.join(test_data_folder_datasets),  # ignore_cache=True
)

derivator.derive_data("pressures", "precision", [0.1])

derivator.derive_data("pressures", "downsample", [5])

derivator.derive_data("pressures", "sensitivity", [
    {"value": 0.1, "shift": "middle"},
])

derivator.derive_data("flows", "precision", [0.1])
derivator.derive_data("flows", "downsample", [5])
derivator.derive_data("flows", "sensitivity", [
    {"value": 0.1, "shift": "middle"},
])

allDerivedDatasets = derivator.get_dervived_datasets()

# # Precision
analyzer.compare([datasets[0], allDerivedDatasets[0]], "pressures", True)
# # Downsample
analyzer.compare([datasets[0], allDerivedDatasets[1]], "pressures", True)
# Sensitivity
analyzer.compare([datasets[0], allDerivedDatasets[2]], "pressures", True)

analyzer.compare([datasets[0], allDerivedDatasets[3]], "flows", True)
analyzer.compare([datasets[0], allDerivedDatasets[4]], "flows", True)
analyzer.compare([datasets[0], allDerivedDatasets[5]], "flows", True)


# %%

dataset = Dataset(os.path.join(test_data_folder_datasets, "graz-ragnitz"))
derivator = DatasetDerivator([dataset], os.path.join(test_data_folder_datasets))

datasets = derivator.derive_model("pipes", "length", "accuracy", [0.01])
datasets[0].id

# %%

# Download
battledim_dataset = DatasetLibrary(test_data_folder_datasets).download(DATASETS.BATTLEDIM)

#%%

generated_dataset_path = os.path.join(test_data_folder_datasets, "synthetic-days-90")
generateDatasetForTimeSpanDays(90, generated_dataset_path)
dataset = Dataset(generated_dataset_path)


#%%



derivator = DatasetDerivator([dataset], os.path.join(test_data_folder_datasets))
# derivedDatasets = derivator.derive_data("demands", "precision", [0.01])
# derivedDatasets = derivator.derive_data("pressures", "precision", [0.01])
derivedDatasets = derivator.derive_data("pressures", "precision", [0.01])
derivedDatasets.append(dataset)

analysis = DatasetAnalyzer(os.path.join(test_data_folder, "out"))

analysis.compare(derivedDatasets, "pressures", quick=True)

#%%

derivator = DatasetDerivator([dataset], os.path.join(test_data_folder_datasets))
derivedDatasets = derivator.derive_data("pressures", "downsample", [60*10])
derivedDatasets.append(dataset)

analysis = DatasetAnalyzer(os.path.join(test_data_folder, "out"))

analysis.compare(derivedDatasets, "pressures", quick=True)

#%%

derivator = DatasetDerivator([dataset], os.path.join(test_data_folder_datasets))
derivedDatasets = derivator.derive_data("pressures", "sensitivity", [0.5])
derivedDatasets.append(dataset)

analysis = DatasetAnalyzer(os.path.join(test_data_folder, "out"))

analysis.compare(derivedDatasets, "pressures")

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
        "synthetic-days-9": {
            "est_length": "10T",
            "C_threshold": 0.01,
            "delta": 0.2,
        }

    }
}

benchmark = LDIMBenchmark(
    hyperparameters,
    battledim_dataset,
    results_dir="./benchmark-results",
    debug=True,
)
benchmark.add_local_methods(local_methods)

# .add_docker_methods(methods)

# %%
# execute benchmark
benchmark.run_benchmark(
    # parallel=True,
)

# %%

benchmark.evaluate(True, write_results=True, generate_plots=True)





# %%
fig, ax = plt.subplots(1)
ax.axvline(1, color="green")
ax.axvspan(
                        0,
                        0.5,
                        color="red",
                        alpha=0.1,
                        lw=0,
                    )
