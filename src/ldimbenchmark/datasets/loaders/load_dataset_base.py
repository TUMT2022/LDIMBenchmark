"""
Module containing the base class for all dataset loaders.
"""
from abc import ABC, abstractmethod


class _LoadDatasetBase(ABC):
    """
    Base class for all dataset loaders.
    """

    @staticmethod
    @abstractmethod
    def download_dataset(download_path=None, force=False):
        """
        Download the dataset from the internet.

        :param download_path: Path to the download folder
        :param force: Force download even if dataset is already downloaded
        """
        pass

    @staticmethod
    @abstractmethod
    def prepare_dataset(unprepared_dataset_path=None, prepared_dataset_path=None):
        """
        Transform the unprepared dataset into a standardized format.

        :param unprepared_dataset_path: Path to the unprepared dataset
        :param prepared_dataset_path: Output path to the prepared dataset
        """
        pass
