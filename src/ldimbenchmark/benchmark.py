from ldimbenchmark.datasets import Dataset, LoadedDataset
from ldimbenchmark.classes import BenchmarkData
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
from typing import Literal, TypedDict, Union, List, Callable
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
from ldimbenchmark.benchmark_complexity import run_benchmark_complexity
from ldimbenchmark.classes import LDIMMethodBase, BenchmarkLeakageResult
import json
import hashlib
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from ldimbenchmark.evaluation import (
    precision,
    recall,
    specifity,
    falsePositiveRate,
    falseNegativeRate,
    f1Score,
)


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
        debug: bool = False,
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
        self.debug = debug
        self.resultsFolder = resultsFolder

    @abstractmethod
    def run(self) -> dict:
        pass


def execute_experiment(experiment: MethodRunner):
    return experiment.run()


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
        # TODO: Rename goal stage method to more meaningful names
        goal: Literal["detection", "location"] = "detection",
        stage="train",  # train, detect
        method="offline",  # offline, online
        debug=False,
        resultsFolder=None,
    ):
        if hyperparameters is None:
            hyperparameters = {}

        for key in hyperparameters.keys():
            matching_params = [
                item
                for item in detection_method.metadata["hyperparameters"]
                if item.get("name") == key
            ]
            # Check if name of the supplied param matches with the ones that can be set
            if len(matching_params) == 0:
                raise Exception(
                    f"Hyperparameter {key} is not known to method {detection_method.name}, must be any of {[param['name'] for param in detection_method.metadata['hyperparameters']]}"
                )
            # Check if the type of the supplied param matches with the ones that can be set
            if not isinstance(hyperparameters[key], matching_params[0].get("type")):
                # Skip Float for now: https://github.com/pandas-dev/pandas/issues/50633
                if isinstance(hyperparameters[key], float):
                    pass
                else:
                    raise Exception(
                        f"Hyperparameter {key}: {hyperparameters[key]} is not of the correct type ({type(hyperparameters[key])}) for method {detection_method.name}, must be any of {[param['type'] for param in detection_method.metadata['hyperparameters'] if param['name'] == key]}"
                    )

        hyperparameter_hash = hashlib.md5(
            json.dumps(hyperparameters, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.id = f"{detection_method.name}_{dataset.id}_{hyperparameter_hash}"
        super().__init__(
            hyperparameters=hyperparameters,
            goal=goal,
            stage=stage,
            method=method,
            resultsFolder=(
                None if resultsFolder == None else os.path.join(resultsFolder, self.id)
            ),
            debug=debug,
        )
        logging.info("Loading Datasets")
        if dataset is str:
            self.dataset = Dataset(dataset).loadDataset().loadBenchmarkData()
        else:
            self.dataset = dataset.loadDataset().loadBenchmarkData()
        self.detection_method = detection_method

    def run(self):

        logging.info(f"Running {self.id} with params {self.hyperparameters}")
        if not self.resultsFolder and self.debug:
            raise Exception("Debug mode requires a results folder.")
        elif self.debug == True:
            additional_output_path = os.path.join(self.resultsFolder, "debug")
            os.makedirs(additional_output_path, exist_ok=True)
        else:
            additional_output_path = None

        # TODO: test compatibility (stages)
        self.detection_method.init_with_benchmark_params(
            additional_output_path=additional_output_path,
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
                self.dataset.evaluation.leaks,
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
                        "dataset_id": self.dataset.id,
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

        return detected_leaks, self.dataset.evaluation.leaks


class LDIMBenchmark:
    def __init__(
        self,
        hyperparameters,
        datasets,
        debug=False,
        results_dir: str = None,
        cache_dir: str = LDIM_BENCHMARK_CACHE_DIR,
    ):
        # validate dataset types and edit them to LoadedDataset
        self.hyperparameters: dict = hyperparameters
        # validate dataset types and edit them to LoadedDataset
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.datasets: List[Dataset] = datasets
        self.experiments: List[MethodRunner] = []
        self.results = {}
        self.cache_dir = cache_dir
        self.results_dir = results_dir
        self.runner_results_dir = os.path.join(self.results_dir, "runner_results")
        self.evaluation_results_dir = os.path.join(
            self.results_dir, "evaluation_results"
        )
        self.complexity_results_dir = os.path.join(
            self.results_dir, "complexity_results"
        )
        self.debug = debug

    # TODO: Make Faster/Inform user about updates
    def add_local_methods(self, methods, goal="detect_offline"):
        """
        Adds local methods to the benchmark.

        :param methods: List of local methods
        """

        if not isinstance(methods, list):
            methods = [methods]
        for dataset in self.datasets:
            for method in methods:
                hyperparameters = None
                if method.name in self.hyperparameters:
                    if dataset.name in self.hyperparameters[method.name]:
                        hyperparameters = self.hyperparameters[method.name][
                            dataset.name
                        ]
                # TODO: Use right hyperparameters
                self.experiments.append(
                    LocalMethodRunner(
                        method,
                        dataset,
                        hyperparameters=hyperparameters,
                        resultsFolder=self.runner_results_dir,
                        debug=self.debug,
                    )
                )

    def add_docker_methods(self, methods: List[str]):
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

    def run_complexity_analysis(
        self,
        methods,
        style: Literal["time", "junctions"],
    ):
        complexity_results_path = os.path.join(self.complexity_results_dir, style)
        os.makedirs(complexity_results_path, exist_ok=True)
        if style == "time":
            return run_benchmark_complexity(
                methods,
                cache_dir=os.path.join(self.cache_dir, "datagen"),
                out_folder=complexity_results_path,
                style="time",
                additionalOutput=self.debug,
            )
        if style == "junctions":
            return run_benchmark_complexity(
                methods,
                cache_dir=os.path.join(self.cache_dir, "datagen"),
                out_folder=complexity_results_path,
                style="junctions",
                additionalOutput=self.debug,
            )

    def run_benchmark(self, parallel=False):
        """
        Runs the benchmark.

        :param parallel: If the benchmark should be run in parallel
        :param results_dir: Directory where the results should be stored
        """
        # TODO: Caching (don't run same experiment twice, if its already there)
        results = []
        if parallel:
            with Pool(processes=cpu_count() - 1) as p:
                max_ = len(self.experiments)
                with tqdm(total=max_) as pbar:
                    for result in p.imap_unordered(
                        execute_experiment, self.experiments
                    ):
                        results.append(result)
                        pbar.update()
            # TODO: preload datasets (as to not overwrite each other during the parallel loop)
            pass
        else:
            for experiment in self.experiments:
                results.append(experiment.run())

    def evaluate(
        self,
        current=True,
        generate_plots=False,
        evaluations: List[Callable] = [
            precision,
            recall,
            specifity,
            falsePositiveRate,
            falseNegativeRate,
            f1Score,
        ],
    ):
        """
        Evaluates the benchmark.

        :param results_dir: Directory where the results are stored
        """
        # TODO: Groupby datasets (and derivations) or by method
        # How does the method perform on different datasets?
        # How do different methods perform on the same dataset?
        # How does one method perform on different derivations of the same dataset?
        # How do different methods perform on one derivations of a dataset?
        # if self.results_dir is None and len(self.results.keys()) == 0:
        #     raise Exception("No results to evaluate")

        # if results_dir:
        #     self.results = self.load_results(results_dir)

        # TODO: Evaluate results
        # TODO: parallelize
        result_folders = glob(os.path.join(self.runner_results_dir, "*"))

        if current:
            result_folders = list(
                filter(
                    lambda x: os.path.basename(x)
                    in [exp.id for exp in self.experiments],
                    result_folders,
                )
            )

        # TODO: Load datasets only once (parallel)
        loaded_datasets = {}
        for dataset in self.datasets:
            loaded = (
                dataset.loadDataset().loadBenchmarkData().getEvaluationBenchmarkData()
            )
            loaded_datasets[dataset.id] = loaded

        results = []
        for experiment_result in [
            os.path.join(result, "") for result in result_folders
        ]:
            detected_leaks = pd.read_csv(
                os.path.join(experiment_result, "detected_leaks.csv"),
                parse_dates=True,
            )

            evaluation_dataset_leakages = pd.read_csv(
                os.path.join(experiment_result, "should_have_detected_leaks.csv"),
                parse_dates=True,
            )

            run_info = pd.read_csv(
                os.path.join(experiment_result, "run_info.csv")
            ).iloc[0]

            # TODO: Ignore Detections outside of the evaluation period
            (evaluation_results, matched_list) = evaluate_leakages(
                evaluation_dataset_leakages, detected_leaks
            )
            evaluation_results["method"] = run_info["method"]
            # TODO: generate name with derivations in brackets
            evaluation_results["dataset"] = run_info["dataset"]
            evaluation_results["dataset_id"] = run_info["dataset_id"]
            results.append(evaluation_results)

            logging.debug(evaluation_results)

            if generate_plots:
                graph_dir = os.path.join(self.evaluation_results_dir, "per_run")
                os.makedirs(graph_dir, exist_ok=True)

                for index, (expected_leak, detected_leak) in enumerate(matched_list):
                    fig, ax = plt.subplots()
                    name = ""
                    data_to_plot = loaded_datasets[run_info["dataset_id"]].pressures

                    if expected_leak is not None:
                        name = str(expected_leak.leak_time_start)
                        boundarys = (
                            expected_leak.leak_time_end - expected_leak.leak_time_start
                        ) / 6
                        mask = (
                            data_to_plot.index
                            >= expected_leak.leak_time_start - boundarys
                        ) & (
                            data_to_plot.index
                            <= expected_leak.leak_time_end + boundarys
                        )

                    if detected_leak is not None:
                        ax.axvline(detected_leak.leak_time_start, color="green")

                    if expected_leak is None and detected_leak is not None:
                        name = str(detected_leak.leak_time_start) + "_fp"
                        boundarys = (data_to_plot.index[-1] - data_to_plot.index[0]) / (
                            data_to_plot.shape[0] / 6
                        )
                        mask = (
                            data_to_plot.index
                            >= detected_leak.leak_time_start - boundarys
                        ) & (
                            data_to_plot.index
                            <= detected_leak.leak_time_start + boundarys
                        )

                    data_to_plot = data_to_plot[mask]
                    data_to_plot.plot(ax=ax, alpha=0.2)
                    debug_folder = os.path.join(experiment_result, "debug/")
                    if os.path.exists(debug_folder):
                        files = glob(debug_folder + "*")
                        for file in files:
                            try:
                                debug_data = pd.read_csv(
                                    file, parse_dates=True, index_col=0
                                )
                                debug_data = debug_data[mask]
                                debug_data.plot(ax=ax, alpha=1)
                            except e:
                                print(e)
                                pass

                    # For some reason the vspan vanishes if we do it earlier so we do it last
                    if expected_leak is not None:
                        ax.axvspan(
                            expected_leak.leak_time_start,
                            expected_leak.leak_time_end,
                            color="red",
                            alpha=0.1,
                            lw=0,
                        )

                    box = ax.get_position()
                    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                    if detected_leak is None and expected_leak is not None:
                        name = str(expected_leak.leak_time_start) + "_fn"

                    # TODO: Plot Leak Outflow, if available

                    # Put a legend to the right of the current axis
                    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                    fig.savefig(os.path.join(graph_dir, name + ".png"))
                    plt.close(fig)
                # TODO: Draw plots with leaks and detected leaks

        results = pd.DataFrame(results)

        for function in evaluations:
            results = function(results)

        # https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262
        results = results.set_index(["method", "dataset_id"])

        os.makedirs(self.evaluation_results_dir, exist_ok=True)

        columns = [
            "TP",
            "FP",
            "TN",
            "FN",
            "TTD",
            "wrongpipe",
            "dataset",
            # "score",
            "precision",
            "recall (TPR)",
            "TNR",
            "FPR",
            "FNR",
            "F1",
        ]
        results.columns = columns

        print(tabulate(results, headers="keys"))
        results.to_csv(os.path.join(self.evaluation_results_dir, "results.csv"))

        results.style.format(escape="latex").set_table_styles(
            [
                # {'selector': 'toprule', 'props': ':hline;'},
                {"selector": "midrule", "props": ":hline;"},
                # {'selector': 'bottomrule', 'props': ':hline;'},
            ],
            overwrite=False,
        ).relabel_index(columns, axis="columns").to_latex(
            os.path.join(self.evaluation_results_dir, "results.tex"),
            position_float="centering",
            clines="all;data",
            column_format="ll|" + "r" * len(columns),
            position="H",
            label="table:benchmark_results",
            caption="Overview of the benchmark results.",
        )
        return results


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
        goal: Literal["detection", "location"] = "detection",
        stage="train",  # train, detect
        method="offline",  # offline, online
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


class FileBasedMethodRunner(MethodRunner):
    def __init__(
        self,
        detection_method: LDIMMethodBase,
        inputFolder: str = "/input",
        outputFolder: str = "/output",
        debug=False,
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
            debug=debug,
        )
        self.detection_method = detection_method
        self.dataset = Dataset(inputFolder).loadDataset().loadBenchmarkData()
        self.id = f"{self.dataset.name}"

    def run(self):
        if not self.resultsFolder and self.debug:
            raise Exception("Debug mode requires a results folder.")
        elif self.debug == True:
            additional_output_path = os.path.join(self.resultsFolder, "debug", "/")
            os.makedirs(additional_output_path, exist_ok=True)
        else:
            additional_output_path = None

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


# TODO: Generate overlaying graphs of leak size and detection times (and additional output)
