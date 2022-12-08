from abc import ABC, abstractmethod


class _LoadDatasetBase(ABC):
    @staticmethod
    @abstractmethod
    def downloadBattledimDataset(downloadPath=None, force=False):
        pass

    @staticmethod
    @abstractmethod
    def prepareBattledimDataset(unpreparedDatasetPath=None, preparedDatasetPath=None):
        pass
