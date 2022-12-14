# %%
%load_ext autoreload
%autoreload 2
from ldimbenchmark.datasets.analysis import DatasetAnalyzer
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark.datasets.derivation import DatasetDerivator
from ldimbenchmark.methods import MNF, LILA

from ldimbenchmark.benchmark import LDIMBenchmark
import logging
from shared import TEST_DATA_FOLDER
import os
from matplotlib import pyplot as plt

# %%

# Download
battledim_dataset = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)


#%%
dataset = Dataset("test_data/datasets/generated/synthetic-days-9")

derivator = DatasetDerivator([dataset], os.path.join(TEST_DATA_FOLDER, "test_derive"))
# derivedDatasets = derivator.derive_data("demands", "noise", [0.01])
# derivedDatasets = derivator.derive_data("pressures", "noise", [0.01])
derivedDatasets = derivator.derive_data("pressures", "noise", [0.01])
derivedDatasets.append(dataset)

analysis = DatasetAnalyzer(os.path.join(TEST_DATA_FOLDER,"out"))

analysis.compare(derivedDatasets)


# analysis.analyze(derivedDatasets)

# %%

logLevel = "INFO"

numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(level=numeric_level, handlers=[logging.StreamHandler()])
logging.getLogger().setLevel(numeric_level)


local_methods = [LILA()]

hyperparameters = {
    "LILA": {
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

benchmark.evaluate(False)





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
