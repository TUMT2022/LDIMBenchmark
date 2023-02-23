from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    LocalMethodRunner,
    DockerMethodRunner,
    FileBasedMethodRunner,
)

import pytest


# Only run locally
@pytest.mark.noci
def test_download():
    DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)
