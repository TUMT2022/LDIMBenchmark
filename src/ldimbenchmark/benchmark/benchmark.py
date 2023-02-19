import itertools
from ldimbenchmark.benchmark.runners import DockerMethodRunner, LocalMethodRunner
from ldimbenchmark.benchmark.runners.BaseMethodRunner import MethodRunner
from ldimbenchmark.datasets import Dataset
from ldimbenchmark.classes import BenchmarkData
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Literal, TypedDict, Union, List, Callable
import os
import time
import logging
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
from ldimbenchmark.evaluation_metrics import (
    precision,
    recall,
    specifity,
    falsePositiveRate,
    falseNegativeRate,
    f1Score,
)


def execute_experiment(experiment: MethodRunner):
    """
    Private method for running an experiment in a separate process.
    """
    return experiment.run()


def get_mask(dataset: pd.DataFrame, start, end, extra_timespan):
    return (dataset.index >= start - extra_timespan) & (
        dataset.index <= end + extra_timespan
    )


def plot_leak(
    data_to_plot: Dict[str, pd.DataFrame],
    leak_pair,
    additional_data_dir,
    out_dir,
):
    fig, ax = plt.subplots()
    name = ""
    expected_leak, detected_leak = leak_pair

    reference_leak = None
    boundarys = None
    if expected_leak is not None:
        name = str(expected_leak.leak_time_start)
        reference_leak = expected_leak
        boundarys = (expected_leak.leak_time_end - expected_leak.leak_time_start) / 6

    if detected_leak is not None:
        ax.axvline(detected_leak.leak_time_start, color="green")

    if expected_leak is None and detected_leak is not None:
        name = str(detected_leak.leak_time_start) + "_fp"
        reference_leak = detected_leak

    for sensor_id, sensor_readings in data_to_plot.items():
        if boundarys == None:
            alternative_boundarys = (
                sensor_readings.index[-1] - sensor_readings.index[0]
            ) / (sensor_readings.shape[0] / 6)
        mask = get_mask(
            sensor_readings,
            reference_leak.leak_time_start,
            reference_leak.leak_time_end
            if reference_leak.leak_time_end != None
            else reference_leak.leak_time_start,
            boundarys if boundarys != None else alternative_boundarys,
        )

        sensor_readings = sensor_readings[mask]
        sensor_readings.plot(ax=ax, alpha=0.2)

        # Plot debug data:
    debug_folder = os.path.join(additional_data_dir, "debug/")
    # TODO: Adjust Mask for each debug data
    if os.path.exists(debug_folder):
        files = glob(debug_folder + "*")
        for file in files:
            try:
                debug_data = pd.read_csv(file, parse_dates=True, index_col=0)
                if boundarys == None:
                    alternative_boundarys = (
                        sensor_readings.index[-1] - sensor_readings.index[0]
                    ) / (sensor_readings.shape[0] / 6)
                mask = get_mask(
                    debug_data,
                    reference_leak.leak_time_start,
                    reference_leak.leak_time_end
                    if reference_leak.leak_time_end != None
                    else reference_leak.leak_time_start,
                    boundarys if boundarys != None else alternative_boundarys,
                )
                debug_data = debug_data[mask]
                debug_data.plot(ax=ax, alpha=1)
            except Exception as e:
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
    fig.savefig(os.path.join(out_dir, name + ".png"))
    plt.close(fig)


# TODO: Draw plots with leaks and detected leaks


class LDIMBenchmark:
    def __init__(
        self,
        hyperparameters,
        datasets,
        debug=False,
        results_dir: str = None,
        cache_dir: str = LDIM_BENCHMARK_CACHE_DIR,
    ):
        self.hyperparameters: dict = hyperparameters
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
                    DockerMethodRunner(
                        method,
                        dataset,
                        self.hyperparameters,
                        resultsFolder=self.runner_results_dir,
                    )
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
            pass
        else:
            for experiment in self.experiments:
                results.append(experiment.run())

    def evaluate(
        self,
        current_only=True,
        write_results=False,
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

        :param current_only: Switch for either evaluating only the current benchmark or incorporate previous runs.
        :param write_results: Write the evaluation results to the results directory.
        :param generate_plots: Generate visual plots during the evaluation, for each leak timeframe.
        :param evaluations: The Evaluation Metrics to be run.
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

        if current_only:
            result_folders = list(
                filter(
                    lambda x: os.path.basename(x)
                    in [exp.id for exp in self.experiments],
                    result_folders,
                )
            )
        if len(result_folders) > 1 and generate_plots:
            logging.warning(
                f"You are generating Plots for {len(result_folders)} results! This will take ages, consider only generating them for the dataset you are interested in."
            )

        loaded_datasets = {}
        for dataset in self.datasets:
            if type(dataset) is str:
                loaded = Dataset(dataset)
            else:
                loaded = dataset

            loaded_datasets[dataset.id] = loaded.loadData()

        results = []
        for experiment_result in [
            os.path.join(result, "") for result in result_folders
        ]:
            detected_leaks = pd.read_csv(
                os.path.join(experiment_result, "detected_leaks.csv"),
                parse_dates=True,
                date_parser=lambda x: pd.to_datetime(x, utc=True),
            )

            evaluation_dataset_leakages = pd.read_csv(
                os.path.join(experiment_result, "should_have_detected_leaks.csv"),
                parse_dates=True,
                date_parser=lambda x: pd.to_datetime(x, utc=True),
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
                logging.info("Generating plots...")
                graph_dir = os.path.join(self.evaluation_results_dir, "per_run")
                os.makedirs(graph_dir, exist_ok=True)

                parallel = True
                if parallel:
                    arguments_list = zip(
                        itertools.repeat(
                            getattr(
                                loaded_datasets[run_info["dataset_id"]], "pressures"
                            ),
                            len(matched_list),
                        ),
                        matched_list,
                        itertools.repeat(experiment_result, len(matched_list)),
                        itertools.repeat(graph_dir, len(matched_list)),
                    )

                    with Pool(processes=cpu_count() - 1) as p:
                        jobs = [
                            p.apply_async(func=plot_leak, args=arguments)
                            for arguments in arguments_list
                        ]
                        for job in tqdm(jobs):
                            job.get()  # wait for jobs being executed

                else:
                    for leak_pair in matched_list:
                        plot_leak(
                            getattr(
                                loaded_datasets[run_info["dataset_id"]], "pressures"
                            ),
                            leak_pair=leak_pair,
                            additional_data_dir=experiment_result,
                            out_dir=graph_dir,
                        )
                    for experiment in self.experiments:
                        results.append(experiment.run())

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

        if write_results:
            print("Writing results to disk")
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


# TODO: Generate overlaying graphs of leak size and detection times (and additional output)
