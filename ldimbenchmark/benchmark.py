from ldimbenchmark.datasets import Dataset, LoadedDataset
from ldimbenchmark.classes import BenchmarkData
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
from typing import Literal, TypedDict
import os
import time
import logging
import docker
import tempfile
import yaml
from ldimbenchmark.constants import LDIM_BENCHMARK_CACHE_DIR
from glob import glob
from ldimbenchmark.benchmark_evaluation import evaluate_leakages
from tabulate import tabulate


class BenchmarkLeakageResult(TypedDict):
    pipe_id: str
    leak_start: datetime
    leak_end: datetime
    leak_peak: datetime
    leak_area: float
    leak_diameter: float
    leak_max_flow: float


class LDIMMethodBase(ABC):
    """
    Skeleton for implementing an instance of a leakage detection method.
    Should implement the following methods:
     - train(): If needed, to train the algorithm
     - detect(): To run the algorithm

    Usage CustomAlgorithm(BenchmarkAlgorithm):
    """

    def __init__(
        self, name: str, version: str, additional_output_path=None, hyperparameters={}
    ):
        """
        Initialize the Leakage Detection Method
        additional_output_path: Path to the output folder of the benchmark. Only use if set.
        """
        self.name = name
        self.version = version
        self.logging = True if additional_output_path != None else False
        self.additional_output_path = additional_output_path
        self.hyperparameters = hyperparameters

    def init_with_benchmark_params(
        self, additional_output_path=None, hyperparameters={}
    ):
        """
        Used for initializing the method in the runner (not needed if run manually).

        :param hyperparameters: Hyperparameters for the method
        :param stages: List of stages that should be executed. Possible stages: "train", "detect", "detect_datapoint"
        :param goal: Goal of the benchmark. Possible goals: "detection", "location"
        :param method: Method that should be executed. Possible methods: "offline", "online"
        """
        self.additional_output_path = additional_output_path
        if not hasattr(self, "hyperparameters"):
            self.hyperparameters = {}
        self.hyperparameters.update(hyperparameters)
        pass

    @abstractmethod
    def train(self, data: BenchmarkData):
        """
        Train the algorithm on Test data (if needed)

        The only metric calculated will be the time your model needs to train.

        The Train Data will be an object (BenchmarkData)

        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def detect(self, data: BenchmarkData) -> list[BenchmarkLeakageResult]:
        """
        Detect Leakages on never before seen data. (BenchmarkData)

        This method should return an array of leakages.

        """
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def detect_datapoint(self, evaluation_data) -> BenchmarkLeakageResult:
        """
        Detect Leakage on never before seen datapoint.
        This method is called multiple times for each datapoint in the evaluation data.
        It is your responsibility to store the new datapoint, if you want to use it for refinining your model.

        The Model will still be initialized by calling the `train()` Method (with the Train Dataset) before.

        This method should return a single BenchmarkLeakageResult or None if there is no leak at this datapoint.
        """
        raise NotImplementedError("Please Implement this method")


class MethodRunner(ABC):
    """
    Runner for a single method and dataset.
    """

    def __init__(
        self,
        hyperparameters: dict,
        goal: str,
        stage: str,
        method: str,
        additional_output_path: str = None,
        resultsFolder: str = None,
    ):
        """
        :param hyperparameters: Hyperparameters for the method
        :param stages: List of stages that should be executed. Possible stages: "train", "detect", "detect_datapoint"
        :param goal: Goal of the benchmark. Possible goals: "detection", "location"
        :param method: Method that should be executed. Possible methods: "offline", "online"
        """
        self.hyperparameters = hyperparameters
        self.goal = goal
        self.stages = stage
        self.method = method
        self.additional_output_path = additional_output_path
        self.resultsFolder = resultsFolder

    @abstractmethod
    def run(self) -> dict:
        pass


class LocalMethodRunner(MethodRunner):
    """
    Runner for a local method.

    Leaves the dataset in prisitine state.
    """

    def __init__(
        self,
        detection_method: LDIMMethodBase,
        dataset: Dataset | str,
        hyperparameters: dict = {},
        # TODO: Rename goal stage method to more meaningful names
        goal: Literal["detection"] | Literal["location"] = "detection",
        stage="train",  # train, detect
        method="offline",  # offline, online
        additional_output_path=None,
        resultsFolder=None,
    ):
        self.id = f"{detection_method.name}_{dataset.name}_parameterTODO"  # TODO: Hash of hyperparameters
        super().__init__(
            hyperparameters=hyperparameters,
            goal=goal,
            stage=stage,
            method=method,
            resultsFolder=os.path.join(resultsFolder, self.id),
            additional_output_path=additional_output_path,
        )
        if dataset is str:
            self.dataset = Dataset(dataset).load()
        else:
            self.dataset = dataset.load()
        self.detection_method = detection_method

    def run(self):

        # test compatibility (stages)
        self.detection_method.init_with_benchmark_params(
            self.additional_output_path,
            self.hyperparameters,
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
        detected_leaks = self.detection_method.detect(
            self.dataset.getEvaluationBenchmarkData()
        )

        end = time.time()
        time_detection = end - start
        logging.info(
            "> Detection time for '"
            + self.detection_method.name
            + "': "
            + str(time_detection)
        )

        if self.resultsFolder:
            os.makedirs(self.resultsFolder, exist_ok=True)
            pd.DataFrame(
                detected_leaks,
                columns=list(BenchmarkLeakageResult.__annotations__.keys()),
            ).to_csv(
                os.path.join(self.resultsFolder, "detected_leaks.csv"),
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )
            pd.DataFrame(
                self.dataset.leaks_evaluation,
                columns=list(BenchmarkLeakageResult.__annotations__.keys()),
            ).to_csv(
                os.path.join(self.resultsFolder, "should_have_detected_leaks.csv"),
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )
            pd.DataFrame(
                [
                    {
                        "method": self.detection_method.name,
                        "dataset": self.dataset.name,
                        "hyperparameters": self.hyperparameters,
                        "goal": self.goal,
                        "stage": self.stages,
                        "train_time": time_training,
                        "detect_time": time_detection,
                    }
                ],
            ).to_csv(
                os.path.join(self.resultsFolder, "run_info.csv"),
                index=False,
                date_format="%Y-%m-%d %H:%M:%S",
            )

        return detected_leaks, self.dataset.leaks_evaluation


class LDIMBenchmark:
    def __init__(
        self,
        hyperparameters,
        datasets,
        results_dir: str = None,
        cache_folder: str = LDIM_BENCHMARK_CACHE_DIR,
    ):
        # validate dataset types and edit them to LoadedDataset
        self.hyperparameters = hyperparameters
        # validate dataset types and edit them to LoadedDataset
        self.datasets = datasets
        self.experiments: list[MethodRunner] = []
        self.results = {}
        self.cache_folder = cache_folder
        self.results_dir = results_dir
        self.runner_results_dir = os.path.join(self.results_dir, "runner_results")
        self.evaluation_results_dir = os.path.join(
            self.results_dir, "evaluation_results"
        )

    def add_local_methods(self, methods, goal="detect_offline"):
        """
        Adds local methods to the benchmark.

        :param methods: List of local methods
        """
        for dataset in self.datasets:
            for method in methods:
                # TODO: Use right hyperparameters
                self.experiments.append(
                    LocalMethodRunner(
                        method,
                        dataset,
                        self.hyperparameters,
                        resultsFolder=self.runner_results_dir,
                    )
                )

    def add_docker_methods(self, methods: list[str]):
        """
        Adds docker methods to the benchmark.

        :param methods: List of docker images (with tag) which run the according method
        """
        for dataset in self.datasets:
            for method in methods:
                # TODO: Use right hyperparameters
                self.experiments.append(
                    DockerMethodRunner(method, dataset, self.hyperparameters)
                )

    def run_benchmark(self, parallel=False):
        """
        Runs the benchmark.

        :param parallel: If the benchmark should be run in parallel
        :param results_dir: Directory where the results should be stored
        """

        results = []
        if parallel:
            # TODO: preload datasets (as to not overwrite each other during the parallel loop)
            pass
        else:
            for experiment in self.experiments:
                results.append(experiment.run())

    def evaluate(self):
        """
        Evaluates the benchmark.

        :param results_dir: Directory where the results are stored
        """
        # if self.results_dir is None and len(self.results.keys()) == 0:
        #     raise Exception("No results to evaluate")

        # if results_dir:
        #     self.results = self.load_results(results_dir)

        # TODO: Evaluate results
        results = []

        for experiment_result in glob(os.path.join(self.runner_results_dir, "*", "")):
            detected_leaks = pd.read_csv(
                os.path.join(experiment_result, "detected_leaks.csv"),
                parse_dates=True,
            )  # .to_dict("records")

            evaluation_dataset_leakages = pd.read_csv(
                os.path.join(experiment_result, "should_have_detected_leaks.csv"),
                parse_dates=True,
            )  # .to_dict("records")

            run_info = pd.read_csv(
                os.path.join(experiment_result, "run_info.csv")
            ).iloc[0]
            # detected_leaks = parse_obj_as(List[BenchmarkLeakageResult], detected_leaks)
            # evaluation_dataset_leakages = parse_obj_as(
            #     List[BenchmarkLeakageResult], evaluation_dataset_leakages
            # )

            # TODO: Ignore Detections outside of the evaluation period
            evaluation_results = evaluate_leakages(
                evaluation_dataset_leakages, detected_leaks
            )
            evaluation_results["method"] = run_info["method"]
            evaluation_results["dataset"] = run_info["dataset"]
            results.append(evaluation_results)

            logging.info(
                f"{len(detected_leaks)} / {len(evaluation_dataset_leakages)} Dataset Leaks"
            )
            logging.info(evaluation_results)

        results = pd.DataFrame(results)
        # https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262
        results["precision"] = results["true_positives"] / (
            results["true_positives"] + results["false_positives"]
        )

        # True-Positive-Rate (Recall)
        results["recall (TPR)"] = results["true_positives"] / (
            results["true_positives"] + results["false_negatives"]
        )
        # True-Negative-Rate (Specificity)
        results["TNR)"] = results["true_negatives"] / (
            results["true_negatives"] + results["false_positives"]
        )
        # False-Positive-Rate (Fall-Out)
        results["FPR"] = results["false_positives"] / (
            results["true_negatives"] + results["false_positives"]
        )
        # False-Negative-Rate (Miss-Rate)
        results["FNR"] = results["false_negatives"] / (
            results["true_positives"] + results["false_negatives"]
        )
        # F1
        results["F1"] = (2 * results["precision"] * results["recall (TPR)"]) / (
            results["precision"] + results["recall (TPR)"]
        )

        print(tabulate(results, headers="keys", showindex="never"))

        os.makedirs(self.evaluation_results_dir, exist_ok=True)
        results.to_csv(os.path.join(self.evaluation_results_dir, "results.csv"))
        results.style.to_latex(os.path.join(self.evaluation_results_dir, "results.tex"))

        pass


class DockerMethodRunner(MethodRunner):
    """
    Runs a leakaged detection method in a docker container.
    """

    # TODO: add support for bind mount parameters? or just define as standard?
    def __init__(
        self,
        image: str,
        dataset: Dataset | str,
        hyperparameters: dict = {},
        goal: Literal["detection"] | Literal["location"] = "detection",
        stage="train",  # train, detect
        method="offline",  # offline, online
        additional_output_path=None,
        resultsFolder=None,
    ):
        super().__init__(
            hyperparameters=hyperparameters,
            goal=goal,
            stage=stage,
            method=method,
            resultsFolder=resultsFolder,
            additional_output_path=additional_output_path,
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


class FileBasedMethodRunner(MethodRunner):
    def __init__(
        self,
        detection_method: LDIMMethodBase,
        inputFolder: str = "/input",
        outputFolder: str = "/output",
        additional_output_path=None,
    ):
        # TODO Read from input Folder
        with open(os.path.join(inputFolder, "options.yml")) as f:
            parameters = yaml.safe_load(f)

        super().__init__(
            hyperparameters=parameters["hyperparameters"],
            goal=parameters["goal"],
            stage=parameters["stage"],
            method=parameters["method"],
            resultsFolder=outputFolder,
            additional_output_path=additional_output_path,
        )
        self.detection_method = detection_method
        self.dataset = Dataset(inputFolder).load()
        self.id = f"{self.dataset.name}"

    def run(self):
        self.detection_method.init_with_benchmark_params(
            self.additional_output_path,
            self.hyperparameters,
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
        detected_leaks = self.detection_method.detect(
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
        return detected_leaks, self.dataset.leaks_evaluation
