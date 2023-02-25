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
        hyperparameters: dict = None,
        goal: Literal[
            "assessment", "detection", "identification", "localization", "control"
        ] = "detection",
        stage: Literal["train", "detect"] = "detect",
        method: Literal["offline", "online"] = "offline",
        debug=False,
        resultsFolder=None,
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
        else:
            self.resultsFolder = os.path.join(resultsFolder, self.id)

        # Do some curtesy checks for LocalMethod Executions
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
                # Skip Float for now: https://github.com/pandas-dev/pandas/issues/50633
                if isinstance(hyperparameters[key], float):
                    pass
                else:
                    raise Exception(
                        f"Hyperparameter {key}: {hyperparameters[key]} is not of the correct type ({type(hyperparameters[key])}) for method {detection_method.name}, must be any of {[param.type for param in detection_method.metadata['hyperparameters'] if param.name == key]}"
                    )

        self.detection_method = detection_method

    def run(self):
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
        start = time.time()

        self.detection_method.train(self.dataset.getTrainingBenchmarkData())
        end = time.time()
        time_training = end - start
        logging.info(
            "> Training time for '"
            + self.detection_method.name
            + "': "
            + str(time_training)
        )

        start = time.time()
        detected_leaks = self.detection_method.detect_offline(
            copy.deepcopy(self.dataset.getEvaluationBenchmarkData())
        )

        end = time.time()
        time_detection = end - start
        logging.info(
            "> Detection time for '"
            + self.detection_method.name
            + "': "
            + str(time_detection)
        )

        self.writeResults(
            detected_leaks=detected_leaks,
            time_training=time_training,
            time_detection=time_detection,
        )

        return self.resultsFolder
