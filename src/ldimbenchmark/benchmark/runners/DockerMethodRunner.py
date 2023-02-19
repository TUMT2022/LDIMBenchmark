from datetime import time
import hashlib
import json
import logging
import os
import tempfile
from typing import Literal, Union
import pandas as pd

import docker
import yaml
from ldimbenchmark.benchmark.runners.BaseMethodRunner import MethodRunner
from ldimbenchmark.classes import BenchmarkLeakageResult, LDIMMethodBase
from ldimbenchmark.datasets.classes import Dataset


# TODO: Probably merge some functionality with LocalMethodRunner as parent class
class DockerMethodRunner(MethodRunner):
    """
    Runs a leakage detection method in a docker container.
    """

    # TODO: add support for bind mount parameters? or just define as standard?
    def __init__(
        self,
        image: str,
        dataset: Union[Dataset, str],
        hyperparameters: dict = {},
        goal: Literal[
            "assessment", "detection", "identification", "localization", "control"
        ] = "detection",
        stage: Literal["train", "detect"] = "detect",
        method: Literal["offline", "online"] = "offline",
        debug=False,
        resultsFolder=None,
    ):
        hyperparameter_hash = hashlib.md5(
            json.dumps(hyperparameters, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.id = f"{image}_{dataset.id}_{hyperparameter_hash}"
        super().__init__(
            hyperparameters=hyperparameters,
            goal=goal,
            stage=stage,
            method=method,
            resultsFolder=None
            if resultsFolder == None
            else os.path.join(resultsFolder, self.id),
            debug=debug,
        )
        self.image = image
        self.dataset = dataset
        self.id = f"{image}_{dataset.name}"

    def run(self):
        folder_parameters = tempfile.TemporaryDirectory()
        path_options = os.path.join(folder_parameters.name, "options.yaml")
        with open(path_options, "w") as f:
            yaml.dump(
                {
                    "hyperparameters": self.hyperparameters,
                    "goal": self.goal,
                    "stage": self.stage,
                    "method": self.method,
                },
                f,
            )

        outputFolder = self.resultsFolder
        if outputFolder is None:
            tempfolder = tempfile.TemporaryDirectory()
            outputFolder = tempfolder.name

        print(outputFolder)
        # download image
        # test compatibility (stages)
        client = docker.from_env()
        # run docker container
        print(
            client.containers.run(
                self.image,
                # ["echo", "hello", "world"],
                volumes={
                    os.path.abspath(self.dataset.path): {
                        "bind": "/input/",
                        "mode": "rw",
                    },
                    path_options: {"bind": "/input/options.yml", "mode": "rw"},
                    os.path.abspath(outputFolder): {"bind": "/output/", "mode": "rw"},
                },
            )
        )
        # mount folder in docker container

        # TODO: Read results from output folder

        detected_leaks = pd.read_csv(
            os.path.join(outputFolder, "detected_leaks.csv"),
            parse_dates=True,
            date_parser=lambda x: pd.to_datetime(x, utc=True),
        ).to_dict("records")
        # if tempfolder:
        #     tempfolder.cleanup()
        # if folder_parameters:
        #     tempfolder.cleanup()
        print(outputFolder)
        return detected_leaks
