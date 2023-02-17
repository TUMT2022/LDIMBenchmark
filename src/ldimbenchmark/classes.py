from pandas import DataFrame
from wntr.network import WaterNetworkModel
from typing import Literal, Optional, TypedDict, Dict, Union, List, Type
from datetime import datetime
from abc import ABC, abstractmethod


class BenchmarkData:
    """
    Representation of the File Based Benchmark Dataset
    """

    def __init__(
        self,
        pressures: Dict[str, DataFrame],
        demands: Dict[str, DataFrame],
        flows: Dict[str, DataFrame],
        levels: Dict[str, DataFrame],
        model: WaterNetworkModel,
        dmas: Dict[str, List[str]],
    ):
        """
        Initialize the BenchmarkData object.
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
        self.dmas = dmas
        """
        District Metered Areas
        Dictionary with names of the areas as key and list of WN nodes as value.
        """
        self.metadata = {}
        """Metadata of the System. e.g. Metering zones and included sensors."""


class BenchmarkLeakageResult(TypedDict):
    leak_pipe_id: Optional[str]
    leak_time_start: datetime
    leak_time_end: datetime
    leak_time_peak: datetime
    leak_area: float
    leak_diameter: float
    leak_max_flow: float


class Hyperparameter:
    """
    Definition of a Hyperparameter for a Leakage Detection Method
    """

    name: str
    type: type
    default: Union[str, int, float, bool]
    description: str
    min: Union[int, float]
    max: Union[int, float]
    options: List[Union[str, int, float]]

    def __init__(
        self,
        name: str,
        description: str,
        type: Type,
        default: Union[int, float, bool],
        options: Optional[List[str]] = None,
        min: Optional[Union[int, float]] = None,
        max: Optional[Union[int, float]] = None,
    ):
        """
        ctor.
        """

        self.name = name
        self.description = description

        # Validation
        self.type = type
        if type != type(default):
            raise ValueError(
                f"Parameter 'default' must be of type {type}, but is of type {type(default)}."
            )

        if isinstance(type, bool):
            if options is not None and (min is not None or max is not None):
                raise ValueError(
                    f"Parameter 'options' and 'min/max cannot be set if using type 'bool'."
                )

        if isinstance(type, str):
            if options is None:
                raise ValueError(
                    f"Parameter 'options' must be set if using type 'str'."
                )

        if isinstance(type, int) or isinstance(type, float):
            if options is None and (min is None or max is None):
                raise ValueError(
                    f"Parameter 'options' or 'min/max' must be set if using type 'int/float'."
                )

        if options is not None and (min is not None or max is not None):
            raise ValueError(
                f"Parameters 'options' and 'min/max' cannot be supplied at the same time."
            )
        self.default = default
        self.options = options
        self.min = min
        self.max = max

    # def __str__(self):
    #     return f"{self.name}: {self.value}"

    # def __repr__(self):
    #     return f"{self.name}: {self.value}"

    # def __eq__(self, other):
    #     return self.name == other.name and self.value == other.value

    # def __hash__(self):
    #     return


class MethodMetadataDataNeeded(TypedDict):
    """
    Describing the necessity of the data for the method.

    necessary - The method needs the data to work, otherwise it would fail.
    optional - The data is not necessary for the method, but its presence would enhance it.
    ignored - The data is not necessary for the method and its presence would not enhance it (simply put it is ignored).

    Depending on what is set for the type of data the

    |Selected Need|Provided by dataset|Result     | Data supplied |
    |:------------|:------------------|-----------|---------------|
    |`necessary`  |yes                |Benchmarked|Yes            |
    |`necessary`  |no                 |Skipped    |No             |
    |`optional`   |yes                |Benchmarked|Yes            |
    |`optional`   |no                 |Benchmarked|No             |
    |`ignored`    |yes                |Benchmarked|No             |
    |`ignored`    |no                 |Benchmarked|No             |
    """

    pressures: Literal["necessary", "optional", "ignored"]
    demands: Literal["necessary", "optional", "ignored"]
    flows: Literal["necessary", "optional", "ignored"]
    levels: Literal["necessary", "optional", "ignored"]
    model: Literal["necessary", "optional", "ignored"]
    structure: Literal["necessary", "optional", "ignored"]


class MethodMetadata(TypedDict):
    data_needed: MethodMetadataDataNeeded
    hyperparameters: List[Hyperparameter]


class LDIMMethodBase(ABC):
    """
    Skeleton for implementing an instance of a leakage detection method.
    Should implement the following methods:
     - train(): If needed, to train the algorithm
     - detect(): To run the algorithm

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
        self.version = version
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
    def train(self, train_data: BenchmarkData) -> None:
        """
        Train the algorithm on Test data (if needed)

        The only metric calculated will be the time your model needs to train.

        The Train Data will be an object (BenchmarkData)

        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def detect_offline(
        self, evaluation_data: BenchmarkData
    ) -> List[BenchmarkLeakageResult]:
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
