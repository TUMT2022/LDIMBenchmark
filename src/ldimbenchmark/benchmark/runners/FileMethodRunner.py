import logging
import os
import time

import yaml
from ldimbenchmark.benchmark.runners.BaseMethodRunner import MethodRunner
from ldimbenchmark.classes import LDIMMethodBase
from ldimbenchmark.datasets.classes import Dataset


class FileBasedMethodRunner(MethodRunner):
    def __init__(
        self,
        detection_method: LDIMMethodBase,
        inputFolder: str = "/input",
        argumentsFolder: str = "/args",
        outputFolder: str = "/output",
    ):
        with open(os.path.join(argumentsFolder, "options.yml")) as f:
            parameters = yaml.safe_load(f)

        super().__init__(
            runner_base_name=detection_method.name,
            dataset=Dataset(inputFolder),
            hyperparameters=parameters["hyperparameters"],
            goal=parameters["goal"],
            stage=parameters["stage"],
            method=parameters["method"],
            resultsFolder=outputFolder,
            debug=parameters["debug"],
        )
        self.detection_method = detection_method
        if self.debug:
            logging.info("Debug logging activated.")

    def run(self) -> str:
        logging.info(f"Running {self.id} with params {self.hyperparameters}")

        self.dataset.loadData()
        self.dataset.loadBenchmarkData()

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
            self.dataset.getEvaluationBenchmarkData()
        )
        end = time.time()

        time_detection = end - start
        logging.info(
            "> Detection time for '"
            + self.detection_method.name
            + "': "
            + str(end - start)
        )

        self.writeResults(
            detected_leaks=detected_leaks,
            time_training=time_training,
            time_detection=time_detection,
        )

        return self.resultsFolder
