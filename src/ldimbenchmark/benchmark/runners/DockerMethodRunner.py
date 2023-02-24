import asyncio
from datetime import time
import hashlib
import io
import itertools
import json
import logging
import os
import tarfile
from pathlib import Path
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
            runner_base_name=image.split("/")[-1].replace(":", "_"),
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
        path_options = os.path.join(folder_parameters.name, "options.yml")
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

        # test compatibility (stages)

        client = docker.from_env()
        if self.docker_base_url != "unix://var/run/docker.sock":
            client = docker.DockerClient(base_url=self.docker_base_url)

        try:
            image = client.images.get(self.image)
        except docker.errors.ImageNotFound:
            logging.info("Image does not exist. Pulling it...")
            client.images.pull(self.image)
        image = client.images.get(self.image)
        wait_script = f"mkdir -p /input/\nmkdir -p /args/\nmkdir -p /output/\nset -e\nwhile [ ! -f /args/options.yml ]; do sleep 1; echo waiting; done\n{' '.join(image.attrs['Config']['Cmd'])}\nls -l /output/\ntar fcz /output.tar  -C / output/\n"
        # run docker container
        try:
            container = client.containers.run(
                self.image,
                [
                    "/bin/sh",
                    "-c",
                    f"printf '{wait_script}' > ./script.sh && chmod +x ./script.sh && ./script.sh",
                ],
                volumes={
                    os.path.abspath(self.dataset.path): {
                        "bind": "/input/",
                        "mode": "ro",
                    }
                },
                mem_limit="12g",
                cpu_count=4,
                detach=True,
            )

            # Prepare Dataset Transfer
            # stream = io.BytesIO()
            # with tarfile.open(fileobj=stream, mode="w|") as tar:
            #     print(self.dataset.path)
            #     files = Path(os.path.join(os.path.abspath(self.dataset.path))).rglob(
            #         "*.*"
            #     )
            #     for file in files:
            #         relative_path = os.path.relpath(
            #             file, os.path.abspath(self.dataset.path)
            #         )
            #         print(relative_path)
            #         with open(file, "rb") as f:
            #             info = tar.gettarinfo(fileobj=f)
            #             info.name = relative_path
            #             tar.addfile(info, f)

            stream_tar_args = io.BytesIO()
            with tarfile.open(fileobj=stream_tar_args, mode="w|") as tar:
                with open(path_options, "rb") as f:
                    info = tar.gettarinfo(fileobj=f)
                    info.name = "options.yml"
                    tar.addfile(info, f)

            container.put_archive("/args/", stream_tar_args.getvalue())
            stream_tar_args.close()

            # TODO: get stats  container.stats(stream=True)
            for log_line in container.logs(stream=True):
                logging.info(f"[{self.image}] {log_line.strip()}")

            # Extract Outputs
            temp_folder_output = tempfile.TemporaryDirectory()
            temp_tar_output = os.path.join(temp_folder_output.name, "output.tar")
            logging.info(temp_tar_output)
            with open(temp_tar_output, "wb") as f:
                # get the bits
                bits, stat = container.get_archive("/output")
                # write the bits
                for chunk in bits:
                    f.write(chunk)

            # unpack
            # logging.info(os.path.abspath(self.resultsFolder))
            def members(tar, strip):
                for member in tar.getmembers():
                    member.path = member.path.split("/", strip)[-1]
                    yield member

            with tarfile.open(temp_tar_output) as tar:
                strip = 1
                tar.extractall(
                    os.path.abspath(self.resultsFolder), members=members(tar, strip)
                )

            container.remove()

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

        # TODO: Write results because we should not include them in the container input
        # self.tryWriteEvaluationLeaks()
        logging.info(f"Results in {self.resultsFolder}")
        return self.resultsFolder
