import logging
import os
import time

import yaml
from ldimbenchmark.benchmark.runners.LocalMethodRunner import LocalMethodRunner
from ldimbenchmark.benchmark.runners.BaseMethodRunner import MethodRunner
from ldimbenchmark.classes import LDIMMethodBase
from ldimbenchmark.datasets.classes import Dataset


class FileBasedMethodRunner(LocalMethodRunner):
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
            detection_method=detection_method,
            dataset=inputFolder,
            dataset_part=parameters["dataset_part"],
            hyperparameters=parameters["hyperparameters"],
            goal=parameters["goal"],
            stage=parameters["stage"],
            method=parameters["method"],
            debug=parameters["debug"],
            resultsFolder=outputFolder,
            createFolder=False,
        )
        if self.debug:
            logging.info("Debug logging activated.")

    def run(self) -> str:
        super().run()
        # start = time.time()
        # logging.info(f"Running {self.id} with params {self.hyperparameters}")

        # self.dataset.loadData()
        # self.dataset.loadBenchmarkData()

        # self.detection_method.init_with_benchmark_params(
        #     additional_output_path=self.additional_output_path,
        #     hyperparameters=self.hyperparameters,
        # )

        # end = time.time()
        # time_initializing = end - start
        # logging.info(
        #     "> Initialization time for '"
        #     + self.detection_method.name
        #     + "': "
        #     + str(time_initializing)
        # )

        # training_data = self.dataset.getTrainingBenchmarkData()
        # start = time.time()
        # if self.dataset_part == "training":
        #     self.detection_method.prepare()
        # elif self.dataset_part == "evaluation":
        #     self.detection_method.prepare(training_data)
        # end = time.time()

        # time_training = end - start
        # logging.info(
        #     "> Training time for '"
        #     + self.detection_method.name
        #     + "': "
        #     + str(time_training)
        # )

        # evaluation_data = self.dataset.getEvaluationBenchmarkData()
        # start = time.time()
        # if self.dataset_part == "training":
        #     detected_leaks = self.detection_method.prepare(training_data)
        # elif self.dataset_part == "evaluation":
        #     detected_leaks = self.detection_method.prepare(evaluation_data)
        # end = time.time()

        # time_detection = end - start
        # logging.info(
        #     "> Detection time for '"
        #     + self.detection_method.name
        #     + "': "
        #     + str(end - start)
        # )

        # self.writeResults(
        #     method_default_hyperparameters=self.detection_method.hyperparameters,
        #     detected_leaks=detected_leaks,
        #     time_training=time_training,
        #     time_detection=time_detection,
        #     time_initializing=time_initializing,
        # )

        # return self.resultsFolder
