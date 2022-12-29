from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    LocalMethodRunner,
    DockerMethodRunner,
    FileBasedMethodRunner,
)


def test_download():
    DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)
