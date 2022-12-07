from pandas import DataFrame
from wntr.network import WaterNetworkModel
from typing import Literal, TypedDict
from datetime import datetime, timedelta


class BenchmarkData:
    """
    Representation of the File Based Bechmark Dataset
    """

    def __init__(
        self,
        pressures: dict[str, DataFrame],
        demands: dict[str, DataFrame],
        flows: dict[str, DataFrame],
        levels: dict[str, DataFrame],
        model: WaterNetworkModel,
    ):
        """
        Hello
        """
        self.pressures = pressures
        """Pressures of the System."""
        self.demands = demands
        """Demands of the System."""
        self.flows = flows
        """Flows of the System."""
        self.levels = levels
        """Levels of the System."""
        self.model = model
        """Model of the System (INP)."""
        self.metadata = {}
        """Metadata of the System. e.g. Metering zones and included sensors."""


class DatasetInfoDatasetOverwrites(TypedDict):
    """
    Dataset Config.yml representation
    """

    file_path: str
    index_column: str
    decimal: str
    delimiter: str


class DatasetInfoDatasetObject(TypedDict):
    """
    Dataset Config.yml representation
    """

    start: datetime
    end: datetime
    overwrites: DatasetInfoDatasetOverwrites


class DatasetInfoDatasetProperty(TypedDict):
    """
    Dataset Config.yml representation
    """

    evaluation: DatasetInfoDatasetObject
    training: DatasetInfoDatasetObject


class DatasetInfo(TypedDict):
    """
    Dataset Config.yml representation
    """

    name: str
    dataset: DatasetInfoDatasetProperty
    inp_file: str
