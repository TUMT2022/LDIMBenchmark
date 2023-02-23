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
        # Overwrite resultsFolder
        if resultsFolder == None:
            self.resultsFolder = None
        else:
            self.resultsFolder = os.path.join(resultsFolder, self.id)

    def run(self):
        logging.info(f"Running {self.id} with params {self.hyperparameters}")
        folder_parameters = tempfile.TemporaryDirectory()
        path_options = os.path.join(folder_parameters.name, "options.yaml")
        with open(path_options, "w") as f:
            yaml.dump(
                {
                    "hyperparameters": self.hyperparameters,
                    "goal": self.goal,
                    "stage": self.stage,
                    "method": self.method,
                    "debug": self.debug,
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
            containerLog = client.containers.run(
                self.image,
                # ["echo", "hello", "world"],
                volumes={
                    os.path.abspath(self.dataset.path): {
                        "bind": "/input/",
                        "mode": "ro",
                    },
                    path_options: {"bind": "/input/options.yml", "mode": "ro"},
                    os.path.abspath(outputFolder): {"bind": "/output/", "mode": "rw"},
                },
                mem_limit="12g",
                cpu_count=4,
            )
            logging.warn(containerLog)

        except docker.errors.ContainerError as e:
            logging.error(f"Method with image {self.image} errored:")
            for line in e.container.logs().decode().split("\n"):
                logging.error(f"Container[{self.image}]: " + line)
            if e.exit_status == 137:
                logging.error("Process in container was killed.")
                logging.error(
                    "This might be due to a memory limit. Try increasing the memory limit."
                )
            return None
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
