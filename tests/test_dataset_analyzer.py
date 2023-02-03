from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark.datasets.analysis import DatasetAnalyzer
from tests.shared import (
    TEST_DATA_FOLDER,
)
import os


def test_analyzer(mocked_dataset1: Dataset):
    analyzer = DatasetAnalyzer(os.path.join(TEST_DATA_FOLDER, "dataset-analysis"))
    analyzer.analyze(mocked_dataset1)
