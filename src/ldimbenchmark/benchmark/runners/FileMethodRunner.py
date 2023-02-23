import logging
import os
import time
import pandas as pd

import yaml
from ldimbenchmark.benchmark.runners.BaseMethodRunner import MethodRunner
from ldimbenchmark.classes import BenchmarkLeakageResult, LDIMMethodBase
from ldimbenchmark.datasets.classes import Dataset


class FileBasedMethodRunner(MethodRunner):
    def __init__(
        self,
        detection_method: LDIMMethodBase,
        inputFolder: str = "/input",
        outputFolder: str = "/output",
        debug=False,
    ):
        with open(os.path.join(inputFolder, "options.yml")) as f:
            parameters = yaml.safe_load(f)

        super().__init__(
            runner_base_name=detection_method.name,
            dataset=Dataset(inputFolder),
            hyperparameters=parameters["hyperparameters"],
            goal=parameters["goal"],
            stage=parameters["stage"],
            method=parameters["method"],
            resultsFolder=outputFolder,
            debug=debug,
        )
        self.detection_method = detection_method

    def run(self):
        if not self.resultsFolder and self.debug:
            raise Exception("Debug mode requires a results folder.")
        elif self.debug == True:
            additional_output_path = os.path.join(self.resultsFolder, "debug", "/")
            os.makedirs(additional_output_path, exist_ok=True)
        else:
            additional_output_path = None

        self.dataset.loadData()
        self.dataset.loadBenchmarkData()

        self.detection_method.init_with_benchmark_params(
            additional_output_path=additional_output_path,
            hyperparameters=self.hyperparameters,
        )

        start = time.time()

        self.detection_method.train(self.dataset.getTrainingBenchmarkData())
        end = time.time()

        logging.info(
            "> Training time for '"
            + self.detection_method.name
            + "': "
            + str(end - start)
        )

        start = time.time()
        detected_leaks = self.detection_method.detect_offline(
            self.dataset.getEvaluationBenchmarkData()
        )

        end = time.time()
        logging.info(
            "> Detection time for '"
            + self.detection_method.name
            + "': "
            + str(end - start)
        )

        pd.DataFrame(
            detected_leaks,
            columns=list(BenchmarkLeakageResult.__annotations__.keys()),
        ).to_csv(
            os.path.join(self.resultsFolder, "detected_leaks.csv"),
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        # TODO write to outputFolder
        return detected_leaks, self.dataset.evaluation.leaks
