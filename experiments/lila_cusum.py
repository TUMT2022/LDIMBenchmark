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
import pandas as pd

from ldimbenchmark.methods.utils.cusum import cusum, cusum_old

test_data_folder = "test_data"
test_data_folder_datasets = os.path.join("test_data", "datasets")

# %%


MRE = pd.read_csv("benchmark-results/runner_results/LILA_synthetic-days-90_99914b932bd37a50b983c5e7c90ae93b/debug/mre.csv", parse_dates=True, index_col="Timestamp")
MRE

# %%

leaks, cusum_data = cusum(
            MRE,
            C_thr=0.01,
            delta=0.2,
            # est_length="10T",
        )

print(leaks)

cusum_data.sum()
# %%
