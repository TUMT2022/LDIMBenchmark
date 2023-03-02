from abc import ABC, abstractmethod
import hashlib
import json
import logging
import os
import time
from typing import Literal, Union

import pandas as pd
from ldimbenchmark.classes import BenchmarkLeakageResult
from ldimbenchmark.datasets.classes import Dataset


class MethodRunner(ABC):
    """
    Runner for a single method and dataset.
    """

    def __init__(
        self,
        runner_base_name: str,
        dataset_part: Union["training", "evaluation"],
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
        self.runner_start_time = time.time()
        if isinstance(dataset, str):
            self.dataset = Dataset(dataset)
        elif isinstance(dataset, Dataset):
            self.dataset = dataset
        else:
            raise Exception(
                f"Method runner argument 'dataset' only takes str or Dataset you provided {type(dataset)}"
            )
        self.hyperparameters = hyperparameters
        if self.hyperparameters is None:
            self.hyperparameters = {}
        hyperparameter_hash = hashlib.md5(
            json.dumps(hyperparameters, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.id = (
            f"{runner_base_name}_{self.dataset.id}_{dataset_part}_{hyperparameter_hash}"
        )

        self.dataset_part = dataset_part
        self.goal = goal
        self.stage = stage
        self.method = method
        self.debug = debug
        self.resultsFolder = resultsFolder

        if not self.resultsFolder and self.debug:
            raise Exception("Debug mode requires a results folder.")
        elif self.debug == True:
            self.additional_output_path = os.path.join(self.resultsFolder, "debug", "")
            os.makedirs(self.additional_output_path, exist_ok=True)
        else:
            self.additional_output_path = None

    @abstractmethod
    def run(self) -> str:
        pass

    def writeResults(
        self,
        method_name,
        method_version,
        method_default_hyperparameters,
        detected_leaks,
        time_training,
        time_detection,
        time_initializing,
    ):
        self.runner_stop_time = time.time()
        if self.resultsFolder:
            os.makedirs(self.resultsFolder, exist_ok=True)
            pd.DataFrame(
                detected_leaks,
                columns=list(BenchmarkLeakageResult.__annotations__.keys()),
            ).to_csv(
                os.path.join(self.resultsFolder, "detected_leaks.csv"),
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )

            pd.DataFrame(
                [
                    {
                        "method": method_name,
                        "method_version": method_version,
                        "dataset": self.dataset.name,
                        "dataset_part": self.dataset_part,
                        "dataset_id": self.dataset.id,
                        "dataset_options": self.dataset.info["derivations"]
                        if "derivations" in self.dataset.info
                        else "{}",
                        "hyperparameters": {
                            **method_default_hyperparameters,
                            **self.hyperparameters,
                        },
                        "goal": self.goal,
                        "stage": self.stage,
                        "train_time": time_training,
                        "detect_time": time_detection,
                        "time_initializing": time_initializing,
                        "total_time": self.runner_stop_time - self.runner_start_time,
                        "executed_at": pd.Timestamp("today").strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                    }
                ],
            ).to_csv(
                os.path.join(self.resultsFolder, "run_info.csv"),
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )
        self.tryWriteEvaluationLeaks()

    def tryWriteEvaluationLeaks(self):
        if hasattr(self.dataset.evaluation, "leaks"):
            # TODO: Probably can be removed?
            if self.dataset_part == "training":
                leaks = self.dataset.train.leaks
            else:
                leaks = self.dataset.evaluation.leaks

            pd.DataFrame(
                leaks,
                columns=list(BenchmarkLeakageResult.__annotations__.keys()),
            ).to_csv(
                os.path.join(self.resultsFolder, "should_have_detected_leaks.csv"),
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )
