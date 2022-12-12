# %%
%load_ext autoreload
%autoreload 2
from ldimbenchmark.datasets.analysis import DatasetAnalyzer
from ldimbenchmark.datasets import Dataset
from ldimbenchmark.datasets.derivation import DatasetDerivator

#%%
dataset = Dataset("tests/test_data/generated/synthetic-days-9")

derivator = DatasetDerivator([dataset], "out_derive")
# derivedDatasets = derivator.derive_data("demands", "noise", [0.01])
# derivedDatasets = derivator.derive_data("pressures", "noise", [0.01])
derivedDatasets = derivator.derive_data("pressures", "noise", [0.01])
derivedDatasets.append(dataset)

analysis = DatasetAnalyzer("out")

analysis.compare(derivedDatasets)

# analysis.analyze(derivedDatasets)

# %%
