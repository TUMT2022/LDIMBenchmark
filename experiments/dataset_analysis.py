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

#%%

generated_dataset_path = os.path.join(test_data_folder_datasets, "gjovik")
dataset = Dataset(generated_dataset_path)


#%%


analysis = DatasetAnalyzer(os.path.join(test_data_folder, "out"))

analysis.analyze(dataset)