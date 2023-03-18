from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

from numpy import timedelta64
from ldimbenchmark.benchmark.runners import DockerMethodRunner, LocalMethodRunner
from ldimbenchmark.benchmark.runners.BaseMethodRunner import MethodRunner
from ldimbenchmark.datasets import Dataset
import pandas as pd
from typing import Dict, Literal, TypedDict, Union, List, Callable
import os
import logging
from ldimbenchmark.constants import LDIM_BENCHMARK_CACHE_DIR
from glob import glob
from ldimbenchmark.benchmark_evaluation import evaluate_leakages
from tabulate import tabulate
from ldimbenchmark.benchmark_complexity import run_benchmark_complexity
import matplotlib.pyplot as plt
import enlighten
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
    dataset: Dataset,
    leak_pair,
    additional_data_dir,
    out_dir,
):
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)
    ax_dataset_flows, ax_dataset_pressures, ax_method = gs.subplots(
        sharex=True, sharey=False
    )
    # fig, ax = plt.subplots()
    name = ""
    expected_leak, detected_leak = leak_pair

    reference_leak = None
    boundaries = None
    if expected_leak is not None:
        name = str(expected_leak["leak_time_start"])
        reference_leak = expected_leak
        boundaries = (
            expected_leak["leak_time_end"] - expected_leak["leak_time_start"]
        ) / 6

        ax_dataset_flows.axvspan(
            expected_leak["leak_time_start"],
            expected_leak["leak_time_end"]
            if not pd.isna(expected_leak["leak_time_end"])
            else expected_leak["leak_time_start"],
            color="red",
            alpha=0.1,
            lw=0,
            zorder=1,
        )
        ax_dataset_pressures.axvspan(
            expected_leak["leak_time_start"],
            expected_leak["leak_time_end"]
            if not pd.isna(expected_leak["leak_time_end"])
            else expected_leak["leak_time_start"],
            color="red",
            alpha=0.1,
            lw=0,
            zorder=1,
        )
        ax_method.axvspan(
            expected_leak["leak_time_start"],
            expected_leak["leak_time_end"]
            if not pd.isna(expected_leak["leak_time_end"])
            else expected_leak["leak_time_start"],
            color="red",
            alpha=0.1,
            lw=0,
            zorder=1,
        )
    # Plot detected leak:
    if detected_leak is not None:
        ax_dataset_flows.axvline(
            detected_leak["leak_time_start"], color="green", zorder=4
        )
        ax_dataset_pressures.axvline(
            detected_leak["leak_time_start"], color="green", zorder=4
        )
        ax_method.axvline(detected_leak["leak_time_start"], color="green", zorder=4)

    #
    if expected_leak is None and detected_leak is not None:
        name = str(detected_leak["leak_time_start"]) + "_fp"
        reference_leak = detected_leak

    # Plot expected leak:
    if detected_leak is None and expected_leak is not None:
        name = str(expected_leak["leak_time_start"]) + "_fn"

    for sensor_id, sensor_readings in getattr(dataset, "pressures").items():
        if boundaries == None:
            # Just use first sensor_readings for all...
            boundaries = (sensor_readings.index[-1] - sensor_readings.index[0]) / (
                sensor_readings.shape[0] / 6
            )
            minimum_boundary = timedelta64(1, "D")
            if boundaries < minimum_boundary:
                boundaries = minimum_boundary
        mask = get_mask(
            sensor_readings,
            reference_leak["leak_time_start"],
            reference_leak["leak_time_end"]
            if not pd.isna(reference_leak["leak_time_end"])
            else reference_leak["leak_time_start"],
            boundaries,
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
        # sensor_readings.plot(ax=ax, alpha=0.2, linestyle="solid", zorder=3)
    for sensor_id, sensor_readings in getattr(dataset, "flows").items():
        if boundaries == None:
            # Just use first sensor_readings for all...
            boundaries = (sensor_readings.index[-1] - sensor_readings.index[0]) / (
                sensor_readings.shape[0] / 6
            )
            minimum_boundary = timedelta64(1, "D")
            if boundaries < minimum_boundary:
                boundaries = minimum_boundary
        mask = get_mask(
            sensor_readings,
            reference_leak["leak_time_start"],
            reference_leak["leak_time_end"]
            if not pd.isna(reference_leak["leak_time_end"])
            else reference_leak["leak_time_start"],
            boundaries,
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
    debug_folder = os.path.join(additional_data_dir, "debug/")
    # TODO: Adjust Mask for each debug data
    if os.path.exists(debug_folder):
        files = glob(debug_folder + "*.csv")
        for file in files:
            try:
                debug_data = pd.read_csv(file, parse_dates=True, index_col=0)
                if boundaries == None:
                    alternative_boundarys = (
                        sensor_readings.index[-1] - sensor_readings.index[0]
                    ) / (sensor_readings.shape[0] / 6)
                mask = get_mask(
                    debug_data,
                    reference_leak["leak_time_start"],
                    reference_leak["leak_time_end"]
                    if not pd.isna(reference_leak["leak_time_end"])
                    else reference_leak["leak_time_start"],
                    boundaries,
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
    # TODO: Plot Leak Outflow, if available

    # Put a legend to the right of the current axis
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle(name)
    fig.savefig(os.path.join(out_dir, name + ".png"))
    plt.close(fig)


def load_result(folder: str) -> Dict:
    folder = os.path.join(folder, "")
    detected_leaks_file = os.path.join(folder, "detected_leaks.csv")
    if not os.path.exists(detected_leaks_file):
        return {}

    detected_leaks = pd.read_csv(
        detected_leaks_file,
        parse_dates=True,
        date_parser=lambda x: pd.to_datetime(x, utc=True),
    )

    evaluation_dataset_leakages = pd.read_csv(
        os.path.join(folder, "should_have_detected_leaks.csv"),
        parse_dates=True,
        date_parser=lambda x: pd.to_datetime(x, utc=True),
    )

    run_info = pd.read_csv(os.path.join(folder, "run_info.csv")).iloc[0]

    # TODO: Ignore Detections outside of the evaluation period
    (evaluation_results, matched_list) = evaluate_leakages(
        evaluation_dataset_leakages, detected_leaks
    )
    evaluation_results["method"] = run_info["method"]
    evaluation_results["method_version"] = run_info.get("method_version", None)
    evaluation_results["dataset"] = run_info["dataset"]
    evaluation_results["dataset_part"] = run_info.get("dataset_part", None)
    evaluation_results["dataset_id"] = run_info["dataset_id"]
    evaluation_results["dataset_derivations"] = run_info["dataset_options"]
    evaluation_results["hyperparameters"] = run_info["hyperparameters"]
    evaluation_results["matched_leaks_list"] = matched_list
    evaluation_results["_folder"] = os.path.basename(os.path.dirname(folder))
    evaluation_results["executed_at"] = run_info.get("executed_at", None)

    return evaluation_results


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

    def run_benchmark(
        self,
        evaluation_mode: Union["training", "evaluation"],
        use_cached=True,
        parallel=False,
        parallel_max_workers=0,
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
                dmethod.split(":")[0].split("/")[-1] for dmethod in self.methods_docker
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
            manager.status_bar(
                " Using cached experiments! ",
                position=1,
                fill="-",
                justify=enlighten.Justify.CENTER,
            )
        results = []
        bar_experiments = manager.counter(
            total=num_experiments,
            desc="Experiments",
            unit="experiments",
            count=num_experiments - len(self.experiments),
        )
        bar_experiments.refresh()
        # This line makes sure we can call update with an effect
        if parallel:
            worker_num = os.cpu_count() - 1
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
                results.append(experiment.run())
                bar_experiments.update()
        manager.stop()

    def evaluate(
        self,
        current_only=True,
        resultFilter: Callable = lambda r: r,
        write_results: Union[None, Literal["csv", "db"]] = None,
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

        manager = enlighten.get_manager()
        pbar1 = manager.counter(
            total=len(result_folders),
            desc="Loading Results",
            unit="results",
        )
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
        pbar1.close()

        results = pd.DataFrame(results)

        for function in evaluations:
            results = function(results)

        results = resultFilter(results)
        # https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262
        results = results.set_index(["_folder"])

        os.makedirs(self.evaluation_results_dir, exist_ok=True)

        if write_results == "csv":
            results = results.drop(columns=["matched_leaks_list"])
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
        elif write_results == "db":
            print("Writing results to database")
            result_db = os.path.join(self.evaluation_results_dir, "results.db")
            if os.path.exists(result_db):
                os.remove(result_db)
            engine = create_engine(f"sqlite:///{result_db}")
            try:
                for index, values in results.iterrows():
                    leak_pairs = pd.DataFrame(values["matched_leaks_list"])
                    leak_pairs = pd.concat(
                        [
                            pd.json_normalize(leak_pairs[0]).add_prefix("expected."),
                            (pd.json_normalize(leak_pairs[1]).add_prefix("detected.")),
                        ],
                        axis=1,
                    )
                    leak_pairs = leak_pairs.drop(
                        columns=["expected.type", "detected.type"],
                        errors="ignore",
                    )
                    leak_pairs["result_id"] = index
                    leak_pairs.to_sql("leak_pairs", engine, if_exists="append")
            except Exception as e:
                print(e)
                print(f"Could not write leak pairs to database. For {index}")
            results = results.drop(columns=["matched_leaks_list"])
            results.to_sql("results", engine, if_exists="replace")

        results = results.set_index(["method", "method_version", "dataset_id"])
        results = results.sort_values(by=["F1"])
        # Display in console
        results = results.drop(
            columns=["_folder", "matched_leaks_list"], errors="ignore"
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
        results.columns = columns

        print(tabulate(results, headers="keys"))
        return results

    def evaluate_run(
        self,
        run_id: str,
    ):
        result_folder = os.path.join(self.runner_results_dir, run_id)
        result = load_result(result_folder)

        logging.info("Generating plots per leak ...")
        manager = enlighten.get_manager()
        loaded_datasets = {}
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

        pbar2 = manager.counter(
            total=len(result["matched_leaks_list"]),
            desc="Graphs:",
            unit="graphs",
        )
        parallel = True
        if parallel:
            with ProcessPoolExecutor() as executor:
                # submit all tasks and get future objects
                futures = []
                for leak_pair in result["matched_leaks_list"]:
                    future = executor.submit(
                        plot_leak,
                        loaded_datasets[result["dataset_id"]],
                        leak_pair=leak_pair,
                        additional_data_dir=result_folder,
                        out_dir=graph_dir,
                    )
                    futures.append(future)

                # process results from tasks in order of task completion
                for future in as_completed(futures):
                    future.result()
                    pbar2.update()

        else:
            for leak_pair in result["matched_leaks_list"]:
                plot_leak(
                    getattr(loaded_datasets[result["dataset_id"]], "pressures"),
                    leak_pair=leak_pair,
                    additional_data_dir=result_folder,
                    out_dir=graph_dir,
                )
                pbar2.update()
        pbar2.close()
        manager.stop()

        # Generate Leak Overview
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_axisbelow(True)
        ax.grid(visible=True, axis="x")
        fig.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.gca().xaxis.set_major_locator(mdates.DayLocator(interval=14))

        for index, (expected_leak, detected_leak) in enumerate(
            result["matched_leaks_list"]
        ):
            num = len(result["matched_leaks_list"]) - index
            if expected_leak is not None:
                length = mdates.date2num(
                    expected_leak["leak_time_end"]
                ) - mdates.date2num(expected_leak["leak_time_start"])
                if length <= 0:
                    length = 1
                ax.barh(
                    [num],
                    # [2],
                    [length],
                    left=[mdates.date2num(expected_leak["leak_time_start"])],
                    # height=[100000],
                    label="expected",
                    color="yellow",
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
                    # height=[100000],
                    label="detected",
                    color="green",
                )
        ax.set_yticks(range(1, len(result["matched_leaks_list"]) + 1))
        # ax.xaxis_date()
        ax.set_title("Leak overview")
        ax.set_ylabel("Leaks")
        ax.set_xlabel("Time")
        yellow_patch = patches.Patch(color="yellow", label="Expected Leaks")
        green_patch = patches.Patch(color="green", label="Detected Leaks")
        plt.legend(handles=[yellow_patch, green_patch])
        plt.gcf().autofmt_xdate()
        fig.savefig(os.path.join(graph_dir, "leaks_overview.png"))
        plt.close(fig)

        statistics_table = []

        pd.DataFrame(statistics_table)

        # TODO: Statistics about the leaks
        # Which leaks are detected? short/long, which leaks are not detected?


# TODO: Generate overlaying graphs of leak size and detection times (and additional output)
