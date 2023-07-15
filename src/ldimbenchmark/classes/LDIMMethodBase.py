import logging
import re
from typing import List
from abc import ABC, abstractmethod
from ldimbenchmark.classes.BenchmarkData import BenchmarkData
from ldimbenchmark.classes.BenchmarkLeakageResult import BenchmarkLeakageResult

from ldimbenchmark.classes.MethodMetadata import MethodMetadata


class LDIMMethodBase(ABC):
    """
    Skeleton for implementing an instance of a leakage detection method.
    Should implement the following methods:
     - prepare(): If needed, to train the algorithm
     - detect_online(): To run the algorithm
     - detect_offline(): To run the algorithm

    Usage CustomAlgorithm(BenchmarkAlgorithm):
    """

    def __init__(
        self,
        name: str,
        version: str,
        metadata: MethodMetadata,
        additional_output_path=None,
    ):
        """
        Initialize the Leakage Detection Method
        additional_output_path: Path to the output folder of the benchmark. Only use if set.
        """
        self.name = name
        # Warn in name is not lowercase
        if self.name != self.name.lower():
            logging.warning(
                f"Method name '{self.name}' is not lowercase. This is not recommended."
            )

        self.version = version
        if bool(re.compile(r"[^A-z0-9\.\-]").search(self.version)):
            logging.warning(
                f"Method version contains not allowed characters. Only [A-z0-9] and . or - are allowed."
            )
        self.metadata = metadata
        self.debug = True if additional_output_path != None else False
        self.additional_output_path = additional_output_path
        self.hyperparameters = {}
        for hyperparameter in metadata["hyperparameters"]:
            self.hyperparameters[hyperparameter.name] = hyperparameter.default

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
        self.debug = True if additional_output_path is not None else False
        if not hasattr(self, "hyperparameters"):
            self.hyperparameters = {}
        self.hyperparameters.update(hyperparameters)

    @abstractmethod
    def prepare(self, training_data: BenchmarkData = None) -> None:
        """
        Prepare your method for the detection phase.
        Called once before detect_online or detect_offline.

        This Method should be used to modify the method as to perform best on future data.
        This can include fitting the model to the training data.

        Please note that `training_data` might not be supplied (e.g. if the dataset does not contain training data).


        This method can be used for methods that need to fit to the data before detecting future leaks.
        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def detect_offline(self, data: BenchmarkData) -> List[BenchmarkLeakageResult]:
        """
        Detect Leakage in an "offline" (historical) manner.
        Detect Leakages on never before seen data. (BenchmarkData)

        This method should return an array of leakages.

        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def detect_online(self, evaluation_data) -> BenchmarkLeakageResult:
        """
        Detect Leakage in an "online" (real-time) manner.
        This method is called multiple times for each data point in the evaluation data.
        It is your responsibility to store the new data point, if you want to use it for refining your model.

        The Model will still be initialized by calling the `train()` Method (with the Train Dataset) before.

        This method should return a single BenchmarkLeakageResult or None if there is no leak at this data point.
        """
        raise NotImplementedError("Please Implement this method")
