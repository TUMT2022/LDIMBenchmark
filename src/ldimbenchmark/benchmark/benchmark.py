from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

from numpy import timedelta64
from ldimbenchmark.benchmark.results import load_result
from ldimbenchmark.benchmark.runners import DockerMethodRunner, LocalMethodRunner
from ldimbenchmark.benchmark.runners.BaseMethodRunner import MethodRunner
from ldimbenchmark.datasets import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Literal, TypedDict, Union, List, Callable
import os
import logging
from ldimbenchmark.constants import CPU_COUNT, LDIM_BENCHMARK_CACHE_DIR
from glob import glob
from tabulate import tabulate
from ldimbenchmark.benchmark_complexity import run_benchmark_complexity
import matplotlib.pyplot as plt
import enlighten
from ldimbenchmark.evaluation.sensitivity import evaluate_derivations
from ldimbenchmark.evaluation_metrics import (
    precision,
    recall,
    specifity,
    falsePositiveRate,
    falseNegativeRate,
    f1Score,
)
from concurrent.futures.process import ProcessPoolExecutor
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import patches
import ast
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from ldimbenchmark.utilities import get_method_name_from_docker_image


def execute_experiment(experiment: MethodRunner):
    """
    Private method for running an experiment in a separate process.
    """
    return experiment.run()


def get_mask(dataset: pd.DataFrame, start, end, extra_timespan):
    return (dataset.index >= start - extra_timespan) & (
        dataset.index <= end + extra_timespan
    )


def get_leak_pair_type(expected_leak, detected_leak):
    # False positive
    if expected_leak is None and detected_leak is not None:
        return "fp"
    # False negative
    if detected_leak is None and expected_leak is not None:
        return "fn"
    # True negative are not existent in this type of analysis.
    # True positive
    return ""


def plot_leak(
    dataset: Dataset,
    leak_pair,
    out_dir,
    additional_data_dir=None,
    boundary_timespan_overwrite: pd.Timedelta = None,
    compare_leaks: bool = True,
):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, hspace=0)
    ax_dataset_flows, ax_dataset_pressures, ax_method = gs.subplots(
        sharex=True, sharey=False
    )

    name = ""
    expected_leak, detected_leak = leak_pair

    boundary = None
    reference_time = None
    leak_time = pd.Timedelta(0)
    if expected_leak is not None:
        expected_leak_start = expected_leak["leak_time_start"]
        expected_leak_end = expected_leak["leak_time_end"]
        name = expected_leak_start.strftime("%Y_%m_%d_%H_%M_%S")

        if pd.isna(expected_leak_end):
            expected_leak_end = expected_leak["leak_time_start"]
        if expected_leak_start != expected_leak_end:
            boundary = (expected_leak_end - expected_leak_start) / 3
        reference_time = expected_leak_start
        leak_time = expected_leak_end - expected_leak_start

        if expected_leak_start == expected_leak_end:
            ax_dataset_flows.axvline(expected_leak_start, color="red", zorder=4)
            ax_dataset_pressures.axvline(expected_leak_start, color="red", zorder=4)
            ax_method.axvline(expected_leak_start, color="red", zorder=4)
        else:
            ax_dataset_flows.axvspan(
                expected_leak_start,
                expected_leak_end,
                color="red",
                alpha=0.1,
                lw=0,
                zorder=1,
            )
            ax_dataset_pressures.axvspan(
                expected_leak_start,
                expected_leak_end,
                color="red",
                alpha=0.1,
                lw=0,
                zorder=1,
            )
            ax_method.axvspan(
                expected_leak_start,
                expected_leak_end,
                color="red",
                alpha=0.1,
                lw=0,
                zorder=1,
            )
        ax_method.text(expected_leak_start, 10, "Expected Leak", rotation=90)

    # Plot detected leak:
    if detected_leak is not None:
        ax_dataset_flows.axvline(
            detected_leak["leak_time_start"], color="green", zorder=4
        )
        ax_dataset_pressures.axvline(
            detected_leak["leak_time_start"], color="green", zorder=4
        )
        ax_method.axvline(detected_leak["leak_time_start"], color="green", zorder=4)
        ax_method.text(
            detected_leak["leak_time_start"], 0, "Detected Leak", rotation=90
        )

    #
    if expected_leak is None and detected_leak is not None:
        name = detected_leak["leak_time_start"].strftime("%Y_%m_%d_%H_%M_%S") + "_fp"
        reference_time = detected_leak["leak_time_start"]

    # Plot expected leak:
    if detected_leak is None and expected_leak is not None:
        name = expected_leak["leak_time_start"].strftime("%Y_%m_%d_%H_%M_%S") + (
            "_fn" if compare_leaks else ""
        )

    ax_dataset_pressures.set_ylabel("Pressure")
    for sensor_id, sensor_readings in getattr(dataset, "pressures").items():
        if boundary == None:
            # Just use first sensor_readings for all...
            boundary = (sensor_readings.index[-1] - sensor_readings.index[0]) / (
                sensor_readings.shape[0] / 6
            )
            minimum_boundary = timedelta64(1, "D")
            if boundary < minimum_boundary:
                boundary = minimum_boundary
        if boundary_timespan_overwrite is not None:
            boundary = boundary_timespan_overwrite
        mask = get_mask(
            sensor_readings,
            reference_time,
            reference_time + leak_time,
            boundary,
        )

        sensor_readings = sensor_readings[mask]
        # Do not use df.plot(): https://github.com/pandas-dev/pandas/issues/51795
        ax_dataset_pressures.plot(
            sensor_readings.index,
            sensor_readings[sensor_id],
            alpha=0.2,
            linestyle="solid",
            zorder=3,
            label=sensor_id,
        )
    ax_dataset_flows.set_ylabel("Flow")
    for sensor_id, sensor_readings in getattr(dataset, "flows").items():
        if boundary == None:
            # Just use first sensor_readings for all...
            boundary = (sensor_readings.index[-1] - sensor_readings.index[0]) / (
                sensor_readings.shape[0] / 6
            )
            minimum_boundary = timedelta64(1, "D")
            if boundary < minimum_boundary:
                boundary = minimum_boundary
        if boundary_timespan_overwrite is not None:
            boundary = boundary_timespan_overwrite
        mask = get_mask(
            sensor_readings,
            reference_time,
            reference_time + leak_time,
            boundary,
        )

        sensor_readings = sensor_readings[mask]
        # Do not use df.plot(): https://github.com/pandas-dev/pandas/issues/51795
        ax_dataset_flows.plot(
            sensor_readings.index,
            sensor_readings[sensor_id],
            alpha=0.2,
            linestyle="solid",
            zorder=3,
            label=sensor_id,
        )

    # Plot debug data:
    if additional_data_dir is not None:
        debug_folder = os.path.join(additional_data_dir, "debug/")
        # TODO: Adjust Mask for each debug data
        ax_method.set_ylabel("Debug")
        ax_method.set_xlabel("Time")
        if os.path.exists(debug_folder):
            files = glob(debug_folder + "*.csv")
            for file in files:
                try:
                    debug_data = pd.read_csv(file, parse_dates=True, index_col=0)
                    if boundary == None:
                        alternative_boundarys = (
                            sensor_readings.index[-1] - sensor_readings.index[0]
                        ) / (sensor_readings.shape[0] / 6)
                    if boundary_timespan_overwrite is not None:
                        boundary = boundary_timespan_overwrite
                    mask = get_mask(
                        debug_data,
                        reference_time,
                        reference_time + leak_time,
                        boundary,
                    )
                    debug_data = debug_data[mask]
                    for column in debug_data.columns:
                        ax_method.plot(
                            debug_data.index,
                            debug_data[column],
                            alpha=1,
                            linestyle="dashed",
                            zorder=3,
                            label=column,
                        )
                    # Do not use df.plot(): https://github.com/pandas-dev/pandas/issues/51795
                    # debug_data.plot(ax=ax_method, alpha=1, linestyle="dashed", zorder=3)
                except Exception as e:
                    logging.exception(e)
                    pass

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax_method.set_title("Debug Data from Method", y=1.0, pad=-14)
    ax_dataset_pressures.set_title("Pressure Data From Dataset", y=1.0, pad=-14)
    ax_dataset_flows.set_title("Flows Data From Dataset", y=1.0, pad=-14)

    date_form = mdates.DateFormatter("%Y-%m-%d %H:%M")
    ax_method.xaxis.set_major_formatter(date_form)
    # TODO: Plot Leak Outflow, if available

    # Put a legend to the right of the current axis
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(name)
    fig.savefig(os.path.join(out_dir, name + ".png"))
    plt.close(fig)


def create_plots(
    results: pd.DataFrame,
    method: str,
    dataset: str,
    hyperparameters: List[str],
    performance_metric: str,
    out_folder,
):
    plot_out_folder = os.path.join(out_folder)  # , method, dataset)
    os.makedirs(plot_out_folder, exist_ok=True)
    # max_metric = results[performance_metric].max()
    # min_metric = results[performance_metric].min()

    plot_data = results[(results["method"] == method) & (results["dataset"] == dataset)]
    hyperparameters = list(map(lambda x: "hyperparameters." + x, hyperparameters))

    hyperparameter_combination = list(itertools.combinations(hyperparameters, 2))

    cmap = sns.cm.rocket_r
    fig, axs = plt.subplots(
        ncols=len(hyperparameters) - 1,
        nrows=len(hyperparameters) - 1,
        figsize=(len(hyperparameters) * 4, len(hyperparameters) * 4),
        squeeze=False,
    )
    for row, param_1 in enumerate(hyperparameters):
        for col, param_2 in enumerate(hyperparameters):
            if (param_2, param_1) in hyperparameter_combination:
                real_row = row - 1
                pvt = pd.pivot_table(
                    plot_data,
                    values=performance_metric,
                    index=param_1,
                    columns=param_2,
                )
                if len(pvt) != 0:
                    sns.heatmap(
                        pvt,
                        ax=axs[real_row, col],
                        cmap=cmap,
                        # vmin=min_metric,
                        # vmax=max_metric,
                    )
                axs[real_row, col].set_ylabel(param_1)
                axs[real_row, col].set_xlabel(param_2)
                # axs[real_row, col].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
                # axs[real_row, col].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            else:
                if len(hyperparameters) > 2 and (
                    row + (len(hyperparameters) - col) < (len(hyperparameters) - 1)
                ):
                    axs[row, col - 1].set_axis_off()
    fig.suptitle(f"Heatmaps for {method}-{dataset}", fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(os.path.join(plot_out_folder, f"heatmap_{method}_{dataset}.png"))
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
        multi_parameters: bool = False,
    ):
        """
        Bechmark for leakage detection methods.

        ====================  ========================================================
        **Argument**          **Description**
        --------------------  --------------------------------------------------------
        hyperparameters       A dictionary of hyperparameters for the benchmark.
        datasets              A list of datasets to be used for the benchmark.
        debug                 A boolean indicating whether to run the benchmark in
                                debug mode. If True, the benchmark will run in debug
                                mode. Default is False.
        results_dir           A string indicating the directory where the results
                                should be stored. If None, the results won't be
                                stored. Default is None.
        cache_dir             A string indicating the directory where the cache
                                should be stored. Default is
                                LDIM_BENCHMARK_CACHE_DIR.
        grid_search           A boolean indicating whether the hyperparameters should
                                 be given as lists to run the algorithms with
                                 multiple variations of the parameters.
                                If True, the product of the given hyperparameters
                                will be calculated and the algorithms will be run
                                with all of theses parameters. Default is False.


        """
        self.hyperparameters: dict = hyperparameters
        if not isinstance(datasets, list):
            datasets = [datasets]
        for index, data in enumerate(datasets):
            if isinstance(data, str):
                datasets[index] = Dataset(data)
        self.datasets: List[Dataset] = datasets
        self.experiments: List[MethodRunner] = []
        self.results = {}
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.results_dir = results_dir
        self.runner_results_dir = os.path.join(self.results_dir, "runner_results")
        self.evaluation_results_dir = os.path.join(
            self.results_dir, "evaluation_results"
        )
        self.complexity_results_dir = os.path.join(
            self.results_dir, "complexity_results"
        )
        self.debug = debug
        self.multi_parameters = multi_parameters
        self.methods_docker = []
        self.methods_local = []

    @staticmethod
    def _get_hyperparameters_for_methods_and_datasets(
        method_ids: List[str], dataset_base_ids: List[str], hyperparameters
    ) -> Dict[str, Dict]:
        """ """

        ######
        # Map Method level
        ######

        hyperparameters_method_map = {}
        # If any method is specified in the hyperparameters
        if bool(set(method_ids) & set(hyperparameters.keys())):
            # hyperparameters_without_methods = hyperparameters.copy()
            # for method_id in list(set(method_ids) & set(hyperparameters.keys())):
            #     del hyperparameters_without_methods[method_id]

            for method_id in method_ids:
                hyperparameters_method_map[
                    method_id
                ] = {}  # hyperparameters_without_methods
                if method_id in hyperparameters:
                    hyperparameters_method_map[method_id] = hyperparameters[method_id]

                    # **hyperparameters_without_methods,

        # If any dataset is specified in the hyperparameters
        elif bool(set(dataset_base_ids) & set(hyperparameters.keys())):
            for dataset_base_id in dataset_base_ids:
                if dataset_base_id in hyperparameters:
                    hyperparameters_method_map[dataset_base_id] = hyperparameters[
                        dataset_base_id
                    ]

        else:
            # If no method or dataset is specified in the hyperparameters use default root values for all methods
            for method_id in method_ids:
                hyperparameters_method_map[method_id] = hyperparameters

        ######
        # Map Dataset level
        ######
        hyperparameters_map = {}
        for method_id in method_ids:
            hyperparameters_map[method_id] = {}
            # Check if any base dataset_Ids match
            if bool(
                set(map(lambda x: x.split("-")[0], dataset_base_ids))
                & set(
                    map(
                        lambda x: x.split("-")[0],
                        hyperparameters_method_map[method_id].keys(),
                    )
                )
            ):
                # hyperparameters_without_datasets = hyperparameters_method_map[
                #     method_id
                # ].copy()
                # for dataset_base_id in list(
                #     set(dataset_base_ids)
                #     & set(hyperparameters_method_map[method_id].keys())
                # ):
                #     del hyperparameters_without_datasets[dataset_base_id]

                for dataset_base_id in dataset_base_ids:
                    hyperparameters_map[method_id][
                        dataset_base_id
                    ] = {}  # hyperparameters_without_datasets
                    for key in sorted(hyperparameters_method_map[method_id].keys()):
                        if dataset_base_id.startswith(key):
                            hyperparameters_map[method_id][
                                dataset_base_id
                            ] = hyperparameters_method_map[method_id][key]
                            # {
                            #     **hyperparameters_without_datasets,
                            #     **hyperparameters_method_map[method_id][key],
                            # }
            else:
                if len(dataset_base_ids) == 0:
                    hyperparameters_map[method_id] = hyperparameters_method_map[
                        method_id
                    ]
                else:
                    for dataset_base_id in dataset_base_ids:
                        hyperparameters_map[method_id][
                            dataset_base_id
                        ] = hyperparameters_method_map[method_id]

        # if method_id in hyperparameters:
        #     if dataset_base_id in hyperparameters[method_id]:
        #         hyperparameters = hyperparameters[method_id][dataset_base_id]
        #     else:
        #         hyperparameters = {
        #             k: v
        #             for k, v in hyperparameters[method_id].items()
        #             if k not in dataset_base_id
        #         }
        return hyperparameters_map

    @staticmethod
    def _get_hyperparameters_matrix_from_hyperparameters_with_list(
        hyperparameters: Dict[str, List[Union[str, int, List]]]
    ):
        if len(hyperparameters.keys()) == 0:
            return [{}]
        index = pd.MultiIndex.from_product(
            hyperparameters.values(), names=hyperparameters.keys()
        )
        param_matrix = pd.DataFrame(index=index).reset_index()

        return param_matrix.to_dict(orient="records")

    def add_local_methods(self, methods):
        """
        Adds local methods to the benchmark.

        :param methods: List of local methods
        """

        if not isinstance(methods, list):
            methods = [methods]
        self.methods_local = self.methods_local + methods

    def add_docker_methods(self, methods: List[str]):
        """
        Adds docker methods to the benchmark.

        :param methods: List of docker images (with tag) which run the according method
        """
        if not isinstance(methods, list):
            methods = [methods]

        self.methods_docker = self.methods_docker + methods

    def run_complexity_analysis(
        self,
        methods,
        style: Literal["time", "junctions"],
        n_repeats=3,
        n_measures=10,
        n_max=91,
    ):
        complexity_results_path = os.path.join(self.complexity_results_dir, style)
        os.makedirs(complexity_results_path, exist_ok=True)
        hyperparameters_map = self._get_hyperparameters_for_methods_and_datasets(
            hyperparameters=self.hyperparameters,
            method_ids=[
                get_method_name_from_docker_image(lmethod) for lmethod in methods
            ],
            dataset_base_ids=[],
        )
        return run_benchmark_complexity(
            methods,
            cache_dir=os.path.join(self.cache_dir, "datagen"),
            out_folder=complexity_results_path,
            style=style,
            additionalOutput=self.debug,
            hyperparameters=hyperparameters_map,
            n_repeats=n_repeats,
            n_measures=n_measures,
            n_max=n_max,
        )

    def run_benchmark(
        self,
        evaluation_mode: Union["training", "evaluation"],
        use_cached=True,
        parallel=False,
        parallel_max_workers=0,
        memory_limit=None,
    ):
        """
        Runs the benchmark.

        :param parallel: If the benchmark should be run in parallel
        :param results_dir: Directory where the results should be stored
                evaluation_mode       A string indicating the mode of the benchmark. If
                                "training", the benchmark will be run in training mode and the training data of a data set will be used.
                                If "evaluation", the benchmark will be run in
                                evaluation mode and the evaluation data of a data set will be used.
                                Default is "training".
        """

        if len(self.methods_docker) > 0 and len(self.methods_local) > 0:
            raise ValueError("Cannot run local and docker methods at the same time")

        logging.info("Starting Benchmark")
        logging.info("Preparing Hyperparameters")
        hyperparameters_map = self._get_hyperparameters_for_methods_and_datasets(
            hyperparameters=self.hyperparameters,
            method_ids=[
                get_method_name_from_docker_image(dmethod)
                for dmethod in self.methods_docker
            ]
            + [lmethod.name for lmethod in self.methods_local],
            dataset_base_ids=[dataset.id for dataset in self.datasets],
        )

        # TODO: Move to parallel step execution step in run_benchmark, but still validate at least once
        for dataset in self.datasets:
            for method in self.methods_docker:
                method_name = method.split(":")[0].split("/")[-1]
                self.hyperparameter_list = [
                    hyperparameters_map[method_name][dataset.id]
                ]
                if self.multi_parameters:
                    self.hyperparameter_list = LDIMBenchmark._get_hyperparameters_matrix_from_hyperparameters_with_list(
                        hyperparameters_map[method_name][dataset.id]
                    )

                logging.info(f"Generating {len(self.hyperparameter_list)} Experiments")
                for hyperparameters in self.hyperparameter_list:
                    self.experiments.append(
                        DockerMethodRunner(
                            method,
                            dataset,
                            evaluation_mode,
                            hyperparameters,
                            resultsFolder=self.runner_results_dir,
                            debug=self.debug,
                            cpu_count=1,
                            mem_limit=memory_limit,
                        )
                    )

            for method in self.methods_local:
                self.hyperparameter_list = [
                    hyperparameters_map[method.name][dataset.id]
                ]
                if self.multi_parameters:
                    self.hyperparameter_list = LDIMBenchmark._get_hyperparameters_matrix_from_hyperparameters_with_list(
                        hyperparameters_map[method.name][dataset.id]
                    )

                logging.info(f"Generating {len(self.hyperparameter_list)} Experiments")
                for hyperparameters in self.hyperparameter_list:
                    self.experiments.append(
                        LocalMethodRunner(
                            detection_method=method,
                            dataset=dataset,
                            dataset_part=evaluation_mode,
                            hyperparameters=hyperparameters,
                            resultsFolder=self.runner_results_dir,
                            debug=self.debug,
                        )
                    )

        # Remove already run experiments
        result_folders = glob(os.path.join(self.runner_results_dir, "*"))
        num_experiments = len(self.experiments)
        self.initial_experiments = self.experiments
        # for experiment in self.experiments:
        #     if experiment.resultsFolder in result_folders:
        #         self.experiments.remove(experiment)
        if use_cached:
            self.experiments = list(
                filter(
                    lambda x: x.resultsFolder not in result_folders, self.experiments
                )
            )
        logging.info(f"Executing {len(self.experiments)} experiments.")
        manager = enlighten.get_manager()
        if len(self.experiments) < num_experiments:
            status_bar = manager.status_bar(
                " Using cached experiments! ",
                position=1,
                fill="-",
                justify=enlighten.Justify.CENTER,
                leave=False,
            )
        bar_experiments = manager.counter(
            total=num_experiments,
            desc="Experiments",
            unit="experiments",
            count=num_experiments - len(self.experiments),
        )
        bar_experiments.refresh()
        # This line makes sure we can call update with an effect
        if parallel:
            worker_num = CPU_COUNT
            if parallel_max_workers > 0:
                worker_num = parallel_max_workers
            try:
                with ProcessPoolExecutor(max_workers=worker_num) as executor:
                    # submit all tasks and get future objects
                    futures = [
                        executor.submit(execute_experiment, runner)
                        for runner in self.experiments
                    ]
                    # process results from tasks in order of task completion
                    for future in as_completed(futures):
                        future.result()
                        bar_experiments.update()
            except KeyboardInterrupt:
                executor.shutdown(wait=False)
                # executor._processes.clear()
                os.kill(os.getpid(), 9)
                manager.stop()
        else:
            for experiment in self.experiments:
                experiment.run()
                bar_experiments.update()
        if "status_bar" in locals():
            status_bar.close()
        bar_experiments.close()
        manager.stop()

    def evaluate_derivations(self):
        sensitivity_results_folder = os.path.join(
            self.evaluation_results_dir, "sensitvity"
        )
        os.makedirs(sensitivity_results_folder, exist_ok=True)
        evaluate_derivations(
            os.path.join(self.evaluation_results_dir, "results.db"),
            sensitivity_results_folder,
        )

    def evaluate(
        self,
        current_only=True,
        resultFilter: Callable = lambda r: r,
        print_results: bool = True,
        write_results: List[Union[None, Literal["csv", "db", "tex"]]] = None,
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
        :param evaluations: The Evaluation Metrics to be run.
        """

        if not isinstance(write_results, list):
            write_results = [write_results]

        # TODO: Groupby datasets (and derivations) or by method
        # How does the method perform on different datasets?
        # How do different methods perform on the same dataset?
        # How does one method perform on different derivations of the same dataset?
        # How do different methods perform on one derivations of a dataset?
        # if self.results_dir is None and len(self.results.keys()) == 0:
        #     raise Exception("No results to evaluate")

        # if results_dir:
        #     self.results = self.load_results(results_dir)

        result_folders = glob(os.path.join(self.runner_results_dir, "*"))
        result_folders_frame = pd.DataFrame(result_folders)
        result_folders_frame["id"] = result_folders_frame[0].apply(
            lambda x: os.path.basename(x)
        )

        if current_only:
            if not hasattr(self, "initial_experiments"):
                logging.warning(
                    "Ignoring current_only switch, since no initial experiments were set. This is probably because 'run_benchmark' was not executed before."
                )
            else:
                experiment_ids = [exp.id for exp in self.initial_experiments]
                result_folders_frame = result_folders_frame[
                    result_folders_frame["id"].isin(experiment_ids)
                ]

                result_folders = list(result_folders_frame[0].values)

        pickle_cache_path = os.path.join(self.cache_dir, "results.pkl")
        if os.path.exists(pickle_cache_path):
            logging.info("Reading results from Cache")
            previous_results = pd.read_pickle(pickle_cache_path)

            result_folders_frame = result_folders_frame[
                ~result_folders_frame["id"].isin(previous_results["_folder"])
            ]
            result_folders = list(result_folders_frame[0].values)
        else:
            previous_results = pd.DataFrame()

        manager = enlighten.get_manager()
        pbar1 = manager.counter(
            total=len(result_folders),
            desc="Loading Results",
            unit="results",
        )
        pbar1.refresh()
        results = []
        parallel = True
        if parallel == True:
            with ProcessPoolExecutor() as executor:
                # submit all tasks and get future objects
                futures = [
                    executor.submit(load_result, folder) for folder in result_folders
                ]
                # process results from tasks in order of task completion
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar1.update()
        else:
            for experiment_result in result_folders:
                results.append(load_result(experiment_result))
                pbar1.update()
        pbar1.close()

        results = pd.concat([previous_results, pd.DataFrame(results)])
        results.to_pickle(pickle_cache_path)

        for function in evaluations:
            results = function(results)

        results = resultFilter(results)
        # https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262
        results = results.set_index(["_folder"])
        results["times_to_detection"] = results["times_to_detection"].astype(str)

        os.makedirs(self.evaluation_results_dir, exist_ok=True)

        if "csv" in write_results:
            results = results.drop(
                columns=["matched_leaks_list", "detected_leaks_frame"]
            )
            logging.info("Writing results as csv")
            results.to_csv(os.path.join(self.evaluation_results_dir, "results.csv"))

        if "db" in write_results:
            logging.info("Writing results to database")
            result_db = os.path.join(self.evaluation_results_dir, "results.db")
            if os.path.exists(result_db):
                os.remove(result_db)
            engine = create_engine(f"sqlite:///{result_db}")
            leak_pairs = pd.concat(list(results["detected_leaks_frame"]))

            leak_pairs.to_sql("leak_pairs", engine, if_exists="replace")
            results = results.drop(
                columns=[
                    "matched_leaks_list",
                    "detected_leaks_frame",
                ]
            )
            results.to_sql("results", engine, if_exists="replace")

        # Generate Heatmaps if multiple parameters are used
        if self.multi_parameters:
            results.hyperparameters = results.hyperparameters.astype("str")
            df_hyperparameters = pd.json_normalize(
                results.hyperparameters.apply(ast.literal_eval)
            ).add_prefix("hyperparameters.")
            df_hyperparameters.index = results.index
            df_hyperparameters
            # results = results.drop(columns=["hyperparameters"])
            flat_results = pd.concat([results, df_hyperparameters], axis=1)

            performance_indicator = "F1"

            methods = [
                get_method_name_from_docker_image(dmethod)
                for dmethod in self.methods_docker
            ] + [lmethod.name for lmethod in self.methods_local]

            hyperparameters_map = self._get_hyperparameters_for_methods_and_datasets(
                hyperparameters=self.hyperparameters,
                method_ids=methods,
                dataset_base_ids=[dataset.id for dataset in self.datasets],
            )

            for method in methods:
                for dataset_id, dataset_name in map(
                    lambda x: (x.id, x.name), self.datasets
                ):
                    logging.info(f"Generating Heatmap for {method} {dataset_name}")
                    create_plots(
                        flat_results,
                        method,
                        dataset_name,
                        hyperparameters_map[method][dataset_id].keys(),
                        performance_indicator,
                        self.evaluation_results_dir,
                    )

        results = results.reset_index("_folder")
        results = results.set_index(["method", "method_version", "dataset_id"])
        results = results.sort_values(by=["F1", "true_positives"])
        # Display in console
        console_display = results.drop(
            columns=[
                "_folder",
                "matched_leaks_list",
                "detected_leaks_frame",
                "times_to_detection",
                "train_time",
                "detect_time",
                "time_initializing",
                "total_time",
                "method_time",
            ],
            errors="ignore",
        )
        # TODO: Automatically add selected metrics
        columns = [
            "TP",
            "FP",
            "TN",
            "FN",
            "TTD",
            "wrongpipe",
            "dataset",
            "dataset_part",
            "dataset_derivations",
            "hyperparameters",
            # "score",
            "executed_at",
            "precision",
            "recall (TPR)",
            "TNR",
            "FPR",
            "FNR",
            "F1",
        ]

        if "tex" in write_results:
            console_display.columns = columns
            console_display.style.format(escape="latex").set_table_styles(
                [
                    # {'selector': 'toprule', 'props': ':hline;'},
                    {"selector": "midrule", "props": ":hline;"},
                    # {'selector': 'bottomrule', 'props': ':hline;'},
                ],
                overwrite=False,
                # TODO: Columns does not exist yet
            ).to_latex(
                os.path.join(self.evaluation_results_dir, "results.tex"),
                position_float="centering",
                clines="all;data",
                column_format="ll|" + "r" * len(columns),
                position="H",
                label="table:benchmark_results",
                caption="Overview of the benchmark results.",
            )
        manager.stop()

        if print_results:
            print(tabulate(console_display, headers="keys"))
        return results

    def evaluate_run(self, run_id: str, boundary_timespan_overwrite=None):
        logging.info(f"Evaluating run {run_id}")
        result_folder = os.path.join(self.runner_results_dir, run_id)
        result = load_result(result_folder)

        manager = enlighten.get_manager()
        loaded_datasets = {}
        # TODO: Load datasets from Ids of the to evaluate run...
        for dataset in self.datasets:
            if type(dataset) is str:
                loaded = Dataset(dataset)
            else:
                loaded = dataset

            loaded_datasets[dataset.id] = loaded.loadData()
            if not hasattr(loaded_datasets[dataset.id], "derivations"):
                loaded_datasets[dataset.name] = loaded_datasets[dataset.id]
        graph_dir = os.path.join(self.evaluation_results_dir, "per_run", run_id)
        os.makedirs(graph_dir, exist_ok=True)

        # Generate Leak Overview
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axisbelow(True)
        ax.grid(visible=True, axis="x")
        # fig.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        # fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))

        labels = [None] * len(result["matched_leaks_list"])
        for index, (expected_leak, detected_leak) in enumerate(
            result["matched_leaks_list"]
        ):
            num = len(result["matched_leaks_list"]) - index
            if expected_leak is not None:
                length = mdates.date2num(
                    expected_leak["leak_time_end"]
                ) - mdates.date2num(expected_leak["leak_time_start"])
                if length <= 0:
                    length = 2
                ax.barh(
                    [num],
                    [length],
                    left=[mdates.date2num(expected_leak["leak_time_start"])],
                    label="expected",
                    color="yellow",
                    alpha=0.5,
                )

            # print(detected_leak)
            if detected_leak is not None:
                length = mdates.date2num(
                    detected_leak["leak_time_end"]
                ) - mdates.date2num(detected_leak["leak_time_start"])
                if length <= 0:
                    length = 1
                ax.barh(
                    [num],
                    [length],
                    left=[mdates.date2num(detected_leak["leak_time_start"])],
                    label="detected",
                    color="green",
                    alpha=0.5,
                )
            leak_pair_type = get_leak_pair_type(expected_leak, detected_leak)
            labels[num - 1] = str(num) + (
                "" if leak_pair_type == "" else f" ({leak_pair_type})"
            )

        ax.set_yticks(range(1, len(result["matched_leaks_list"]) + 1))
        ax.set_yticklabels(labels)
        # ax.xaxis_date()
        ax.set_title("Overview of expected and detected leaks")
        ax.set_ylabel("leaks")
        ax.set_xlabel("time")
        ax.set_xlim(
            loaded_datasets[result["dataset_id"]].info["dataset"]["evaluation"][
                "start"
            ],
            loaded_datasets[result["dataset_id"]].info["dataset"]["evaluation"]["end"],
        )
        yellow_patch = patches.Patch(color="yellow", label="expected leaks")
        green_patch = patches.Patch(color="green", label="detected leaks")
        plt.legend(handles=[yellow_patch, green_patch])
        plt.gcf().autofmt_xdate()
        fig.savefig(os.path.join(graph_dir, "leaks_overview.png"))
        plt.close(fig)

        logging.info("Generating plots per leak ...")
        pbar2 = manager.counter(
            total=len(result["matched_leaks_list"]),
            desc="Graphs:",
            unit="graphs",
        )
        parallel = False
        if parallel:
            with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                # submit all tasks and get future objects
                futures = []
                for leak_pair in result["matched_leaks_list"]:
                    future = executor.submit(
                        plot_leak,
                        loaded_datasets[result["dataset_id"]],
                        leak_pair=leak_pair,
                        additional_data_dir=result_folder,
                        out_dir=graph_dir,
                        boundary_timespan_overwrite=boundary_timespan_overwrite,
                    )
                    futures.append(future)

                # process results from tasks in order of task completion
                for future in as_completed(futures):
                    future.result()
                    pbar2.update()

        else:
            for leak_pair in result["matched_leaks_list"]:
                plot_leak(
                    loaded_datasets[result["dataset_id"]],
                    leak_pair=leak_pair,
                    additional_data_dir=result_folder,
                    out_dir=graph_dir,
                )
                pbar2.update()
        pbar2.close()
        manager.stop()

        statistics_table = []

        pd.DataFrame(statistics_table)

        # TODO: Statistics about the leaks
        # Which leaks are detected? short/long, which leaks are not detected?
