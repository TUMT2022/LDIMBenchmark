from abc import ABC, abstractmethod


class _LoadDatasetBase(ABC):
    @abstractmethod
    def downloadBattledimDataset(downloadPath=None, force=False):
        pass

    @abstractmethod
    def prepareBattledimDataset(unpreparedDatasetPath=None, preparedDatasetPath=None):
        pass
