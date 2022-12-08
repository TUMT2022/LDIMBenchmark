from enum import Enum
import tempfile
from ldimbenchmark.datasets.loaders.load_battledim import BattledimDatasetLoader

import os
from ldimbenchmark.datasets.classes import Dataset
import numpy as np


class DATASETS(Enum):
    """
    Enum of available datasets
    """

    BATTLEDIM = "battledim"


class DatasetLibrary:
    """
    Library of datasets
    """

    def __init__(self, download_path: str):
        self.path = download_path

    def download(self, dataset_name: str | list[str], force=False) -> list[Dataset]:
        """
        Downloads a dataset from the internet

        :param dataset_name: Name of the dataset to download
        :param force: Force download even if dataset is already downloaded
        """
        # TODO: Try to import download module for "dataset_name" and execute it.
        # TODO: Implement force download

        if dataset_name is str:
            dataset_name = [DATASETS[dataset_name.upper()]]
        if isinstance(dataset_name, DATASETS):
            dataset_name = [dataset_name]

        datasets = []

        for dataset in dataset_name:
            print("Downloading dataset: " + dataset.value)
            tempdir = tempfile.TemporaryDirectory()
            dataset_download_path = os.path.join(self.path, dataset.value)
            if os.path.exists(dataset_download_path) and not force:
                print("Dataset already downloaded")
                datasets.append(Dataset(dataset_download_path))
                continue
            if dataset == DATASETS.BATTLEDIM:
                BattledimDatasetLoader.downloadBattledimDataset(tempdir.name)
                BattledimDatasetLoader.prepareBattledimDataset(
                    tempdir.name, dataset_download_path
                )

                datasets.append(Dataset(dataset_download_path))

            tempdir.cleanup()
        return datasets
