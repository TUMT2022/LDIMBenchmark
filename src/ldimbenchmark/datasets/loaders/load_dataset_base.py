from abc import ABC, abstractmethod


class _LoadDatasetBase(ABC):
    @staticmethod
    @abstractmethod
    def download_dataset(download_path=None, force=False):
        pass

    @staticmethod
    @abstractmethod
    def prepare_dataset(unprepared_dataset_path=None, prepared_dataset_path=None):
        pass
