from abc import ABC, abstractmethod
from typing import Literal, Union


class MethodRunner(ABC):
    """
    Runner for a single method and dataset.
    """

    def __init__(
        self,
        hyperparameters: dict,
        goal: Literal[
            "assessment", "detection", "identification", "localization", "control"
        ] = "detection",
        stage: Literal["train", "detect"] = "detect",
        method: Literal["offline", "online"] = "offline",
        debug: bool = False,
        resultsFolder: Union[str, None] = None,
    ):
        """
        Base Class for a Method Runner.


        Parameters
        ----------
        detection_method : LDIMMethodBase
            The LDIM method object.

        dataset : Union[Dataset, str]
            The dataset object or the path to the dataset.

        hyperparameters : dict, optional
            The hyperparameters for the LDIM object, by default None

        goal : Literal[
            "assessment", "detection", "identification", "localization", "control"
        ], optional
            Goal of the benchmark. Possible goals:
            "assessment" - Asses if there are any leaks
            "detection" - Detect leaks and their onset
            "localization" - Detect leaks and their location
            "control" - Detect leaks and fomulate a control strategy
            by default "detection"

        stage : Literal["train", "detect"], optional
            List of stages that should be executed. Possible stages: "train", "detect"

        method : Literal["offline", "online"], optional
            The method of the LDIM object, by default "offline"

        debug : bool, optional
            Whether to print debug information, by default False

        resultsFolder : None, optional
            The path to the results folder, by default None

        """
        self.hyperparameters = hyperparameters
        self.goal = goal
        self.stage = stage
        self.method = method
        self.debug = debug
        self.resultsFolder = resultsFolder

    @abstractmethod
    def run(self) -> dict:
        pass
