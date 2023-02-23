from abc import ABC, abstractmethod
import hashlib
import json
import os
from typing import Literal, Union

from ldimbenchmark.datasets.classes import Dataset


class MethodRunner(ABC):
    """
    Runner for a single method and dataset.
    """

    def __init__(
        self,
        runner_base_name: str,
        dataset: Dataset,
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
        if type(dataset) is str:
            self.dataset = Dataset(dataset)
        else:
            self.dataset = dataset

        self.hyperparameters = hyperparameters
        if self.hyperparameters is None:
            self.hyperparameters = {}
        hyperparameter_hash = hashlib.md5(
            json.dumps(hyperparameters, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.id = f"{runner_base_name}_{dataset.id}_{hyperparameter_hash}"

        self.goal = goal
        self.stage = stage
        self.method = method
        self.debug = debug
        if resultsFolder == None:
            self.resultsFolder = None
        else:
            self.resultsFolder = os.path.join(resultsFolder, self.id)

    @abstractmethod
    def run(self) -> dict:
        pass
