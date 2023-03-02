import copy
import hashlib
import json
import logging
import os
import time
from typing import Literal, Union
import pandas as pd

import yaml
from ldimbenchmark.benchmark.runners.BaseMethodRunner import MethodRunner
from ldimbenchmark.classes import BenchmarkLeakageResult, LDIMMethodBase
from ldimbenchmark.datasets.classes import Dataset


class LocalMethodRunner(MethodRunner):
    """
    Runner for a local method.

    Leaves the dataset in prisitine state.
    """

    def __init__(
        self,
        detection_method: LDIMMethodBase,
        dataset: Union[Dataset, str],
        dataset_part: Union["training", "evaluation"] = "training",
        hyperparameters: dict = None,
        goal: Literal[
            "assessment", "detection", "identification", "localization", "control"
        ] = "detection",
        stage: Literal["train", "detect"] = "detect",
        method: Literal["offline", "online"] = "offline",
        debug=False,
        resultsFolder=None,
        createFolder: bool = True,
    ):
        """Initialize the LocalMethodRunner.

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
            The goal of the LDIM object, by default "detection"

        stage : Literal["train", "detect"], optional
            The stage of the LDIM object, by default "detect"

        method : Literal["offline", "online"], optional
            The method of the LDIM object, by default "offline"

        debug : bool, optional
            Whether to print debug information, by default False

        resultsFolder : None, optional
            The path to the results folder, by default None

        Raises
        ------
        TypeError
            If the dataset is not of type Dataset or str.
        ValueError
            If the dataset is not of type Dataset or str.
        """

        super().__init__(
            runner_base_name=f"{detection_method.name}_{detection_method.version}",
            dataset=dataset,
            dataset_part=dataset_part,
            hyperparameters=hyperparameters,
            goal=goal,
            stage=stage,
            method=method,
            resultsFolder=resultsFolder,
            debug=debug,
        )

        # Overwrite resultsFolder
        if resultsFolder == None:
            self.resultsFolder = None
        elif createFolder:
            self.resultsFolder = os.path.join(resultsFolder, self.id)
        else:
            self.resultsFolder = resultsFolder
        # Do some courtesy checks for LocalMethod Executions
        for key in self.hyperparameters.keys():
            if key.startswith("_"):
                continue
            matching_params = [
                item
                for item in detection_method.metadata["hyperparameters"]
                if item.name == key
            ]
            # Check if name of the supplied param matches with the ones that can be set
            if len(matching_params) == 0:
                raise Exception(
                    f"Hyperparameter {key} is not known to method {detection_method.name}, must be any of {[param.name for param in detection_method.metadata['hyperparameters']]}"
                )
            # Check if the type of the supplied param matches with the ones that can be set
            if not isinstance(hyperparameters[key], matching_params[0].type):
                if (
                    # Ignore int to float conversion
                    (
                        isinstance(hyperparameters[key], int)
                        and matching_params[0].type == float
                    )
                    or
                    # Skip Float for now: https://github.com/pandas-dev/pandas/issues/50633
                    isinstance(hyperparameters[key], float)
                ):
                    pass
                else:
                    raise Exception(
                        f"Hyperparameter {key}: {hyperparameters[key]} is not of the correct type ({type(hyperparameters[key])}) for method {detection_method.name}, must be any of {[param.type for param in detection_method.metadata['hyperparameters'] if param.name == key]}"
                    )
        # Check for mandatory params
        for param in detection_method.metadata["hyperparameters"]:
            if param.required == True and param.name not in hyperparameters.keys():
                raise Exception(
                    f"Hyperparameter '{param.name}' is required, but is not set."
                )

        self.detection_method = detection_method

    def run(self):
        start = time.time()
        logging.info(f"Running {self.id} with params {self.hyperparameters}")

        logging.info(f"LocalMethodRunner - Loading Dataset {self.dataset.id}")
        self.dataset.loadData()
        self.dataset.loadBenchmarkData()
        logging.info("Loading Datasets - FINISH")

        # TODO: test compatibility (stages)
        self.detection_method.init_with_benchmark_params(
            additional_output_path=self.additional_output_path,
            hyperparameters=self.hyperparameters,
        )
        end = time.time()
        time_initializing = end - start
        logging.info(
            "> Initialization time for '"
            + self.detection_method.name
            + "': "
            + str(time_initializing)
        )

        preparation_data = self.dataset.getTrainingBenchmarkData()
        start = time.time()
        if self.dataset_part == "training":
            self.detection_method.prepare()
        elif self.dataset_part == "evaluation":
            self.detection_method.prepare(preparation_data)
        end = time.time()
        time_preparation = end - start
        logging.info(
            "> Preparation time for '"
            + self.detection_method.name
            + "': "
            + str(time_preparation)
        )

        start = time.time()
        evaluation_data = copy.deepcopy(self.dataset.getEvaluationBenchmarkData())
        start = time.time()
        if self.dataset_part == "training":
            detected_leaks = self.detection_method.detect_offline(preparation_data)
        elif self.dataset_part == "evaluation":
            detected_leaks = self.detection_method.detect_offline(evaluation_data)

        end = time.time()
        time_detection = end - start
        logging.info(
            "> Detection time for '"
            + self.detection_method.name
            + "': "
            + str(time_detection)
        )

        self.writeResults(
            method_name=self.detection_method.name,
            method_version=self.detection_method.version,
            method_default_hyperparameters=self.detection_method.hyperparameters,
            detected_leaks=detected_leaks,
            time_training=time_preparation,
            time_detection=time_detection,
            time_initializing=time_initializing,
        )

        return self.resultsFolder
