from pandas import DataFrame
from wntr.network import WaterNetworkModel
from typing import Literal, TypedDict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


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


class BenchmarkLeakageResult(TypedDict):
    leak_pipe_id: str
    leak_time_start: datetime
    leak_time_end: datetime
    leak_time_peak: datetime
    leak_area: float
    leak_diameter: float
    leak_max_flow: float


class LDIMMethodBase(ABC):
    """
    Skeleton for implementing an instance of a leakage detection method.
    Should implement the following methods:
     - train(): If needed, to train the algorithm
     - detect(): To run the algorithm

    Usage CustomAlgorithm(BenchmarkAlgorithm):
    """

    def __init__(
        self, name: str, version: str, additional_output_path=None, hyperparameters={}
    ):
        """
        Initialize the Leakage Detection Method
        additional_output_path: Path to the output folder of the benchmark. Only use if set.
        """
        self.name = name
        self.version = version
        self.logging = True if additional_output_path != None else False
        self.additional_output_path = additional_output_path
        self.hyperparameters = hyperparameters

    def init_with_benchmark_params(
        self, additional_output_path=None, hyperparameters={}
    ):
        """
        Used for initializing the method in the runner (not needed if run manually).

        :param hyperparameters: Hyperparameters for the method
        :param stages: List of stages that should be executed. Possible stages: "train", "detect", "detect_datapoint"
        :param goal: Goal of the benchmark. Possible goals: "detection", "location"
        :param method: Method that should be executed. Possible methods: "offline", "online"
        """
        self.additional_output_path = additional_output_path
        if not hasattr(self, "hyperparameters"):
            self.hyperparameters = {}
        self.hyperparameters.update(hyperparameters)
        pass

    @abstractmethod
    def train(self, data: BenchmarkData):
        """
        Train the algorithm on Test data (if needed)

        The only metric calculated will be the time your model needs to train.

        The Train Data will be an object (BenchmarkData)

        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def detect(self, data: BenchmarkData) -> list[BenchmarkLeakageResult]:
        """
        Detect Leakages on never before seen data. (BenchmarkData)

        This method should return an array of leakages.

        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def detect_datapoint(self, evaluation_data) -> BenchmarkLeakageResult:
        """
        Detect Leakage on never before seen datapoint.
        This method is called multiple times for each datapoint in the evaluation data.
        It is your responsibility to store the new datapoint, if you want to use it for refinining your model.

        The Model will still be initialized by calling the `train()` Method (with the Train Dataset) before.

        This method should return a single BenchmarkLeakageResult or None if there is no leak at this datapoint.
        """
        raise NotImplementedError("Please Implement this method")
