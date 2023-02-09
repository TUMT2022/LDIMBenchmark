from datetime import time
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


class DockerMethodRunner(MethodRunner):
    """
    Runs a leakaged detection method in a docker container.
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
        super().__init__(
            hyperparameters=hyperparameters,
            goal=goal,
            stage=stage,
            method=method,
            resultsFolder=resultsFolder,
            debug=debug,
        )
        self.image = image
        self.dataset = dataset
        self.id = f"{image}_{dataset.name}"

    def run(self):
        outputFolder = self.resultsFolder
        if outputFolder is None:
            tempfolder = tempfile.TemporaryDirectory()
            outputFolder = tempfolder.name
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
                        "mode": "ro",
                    },
                    os.path.abspath(outputFolder): {"bind": "/output/", "mode": "rw"},
                },
            )
        )
        # mount folder in docker container

        # TODO: Read results from output folder

        detected_leaks = pd.read_csv(
            os.path.join(outputFolder, "detected_leaks.csv"),
            parse_dates=True,
        ).to_dict("records")
        # if tempfolder:
        #     tempfolder.cleanup()
        print(outputFolder)
        return detected_leaks
