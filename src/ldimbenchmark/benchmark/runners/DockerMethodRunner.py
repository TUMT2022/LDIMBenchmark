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
        hyperparameters: dict = None,
        goal: Literal[
            "assessment", "detection", "identification", "localization", "control"
        ] = "detection",
        stage: Literal["train", "detect"] = "detect",
        method: Literal["offline", "online"] = "offline",
        debug=False,
        resultsFolder=None,
        docker_base_url="unix://var/run/docker.sock",
    ):
        super().__init__(
            runner_base_name=image,
            dataset=dataset,
            hyperparameters=hyperparameters,
            goal=goal,
            stage=stage,
            method=method,
            resultsFolder=resultsFolder,
            debug=debug,
        )

        self.image = image
        self.docker_base_url = docker_base_url

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

        # Remove Colons from path
        outputFolder = outputFolder.replace(":", "_")
        # download image
        # test compatibility (stages)

        client = docker.from_env()
        if self.docker_base_url != "unix://var/run/docker.sock":
            client = docker.DockerClient(base_url=self.docker_base_url)
        # run docker container
        try:
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
                mem_limit="4g",
                cpu_count=4,
            )
        except docker.errors.ContainerError as e:
            logging.error("ContainerError:")
            for line in e.container.logs().decode().split("\n"):
                logging.error("Container: " + line)
            raise e
        # mount folder in docker container
        # name, dst = dst.split(':')
        #     container = client.containers.get(name)

        #     os.chdir(os.path.dirname(src))
        #     srcname = os.path.basename(src)
        #     tar = tarfile.open(src + '.tar', mode='w')
        #     try:
        #         tar.add(srcname)
        #     finally:
        #         tar.close()

        #     data = open(src + '.tar', 'rb').read()
        #     container.put_archive(os.path.dirname(dst), data)
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
