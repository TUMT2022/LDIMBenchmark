from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import importlib
import os
import shutil
import time
from datetime import datetime
from glob import glob
import logging
import docker

import enlighten
from matplotlib import pyplot as plt
from ldimbenchmark.benchmark.results import load_result
from ldimbenchmark.benchmark.runners import DockerMethodRunner
from ldimbenchmark.constants import LDIM_BENCHMARK_CACHE_DIR
from ldimbenchmark.datasets import Dataset
from ldimbenchmark.generator import (
    generateDatasetsForTimespan,
    generateDatasetsForJunctions,
)
from ldimbenchmark.classes import LDIMMethodBase
import numpy as np
import pandas as pd
import wntr
import yaml
import big_o
import matplotlib as mpl
from typing import List

from ldimbenchmark.utilities import get_method_name_from_docker_image


def loadDataset_local(dataset_path):
    dataset = Dataset(dataset_path)
    dataset.loadData().loadBenchmarkData()
    del dataset
    dataset = Dataset(dataset_path)

    # dataset.is_valid()
    number = int(os.path.basename(os.path.normpath(dataset_path)).split("-")[-1])
    return (
        number,
        dataset
        # dataset.getTrainingBenchmarkData(),
        # dataset.getEvaluationBenchmarkData(),
    )


def run_benchmark_complexity(
    methods: List[str],
    hyperparameters,
    cache_dir=os.path.join(LDIM_BENCHMARK_CACHE_DIR, "datagen"),
    out_folder="out/complexity",
    style=None,
    additionalOutput=False,
    n_repeats=3,
    n_measures=10,
    n_max=91,
):
    """
    Run the benchmark for the given methods and datasets.
    :param methods: List of methods to run (only supports LocalMethodRunner currently)
    """

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    logging.info("Complexity Analysis:")
    logging.info(" > Generating Datasets")
    if style == "time":
        datasets_dir = os.path.join(cache_dir, "synthetic-days")
        generateDatasetsForTimespan(1, n_max, datasets_dir)
    if style == "junctions":
        datasets_dir = os.path.join(cache_dir, "synthetic-junctions")
        generateDatasetsForJunctions(4, n_max, datasets_dir)

    dataset_dirs = glob(datasets_dir + "/*/")
    min_n = 4
    n_samples = np.linspace(min_n, n_max - 1, n_measures).astype("int64")

    manager = enlighten.get_manager()
    bar_loading_data = manager.counter(
        total=len(dataset_dirs), desc="Validating data", unit="datasets"
    )
    bar_loading_data.update(incr=0)

    # logging.info(" > Loading Data")
    datasets = {}
    try:
        parallel = True
        if parallel:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # submit all tasks and get future objects
                futures = [
                    executor.submit(loadDataset_local, dataset_dir)
                    for dataset_dir in dataset_dirs
                ]
                # process results from tasks in order of task completion
                for future in as_completed(futures):
                    dataset_id, dataset = future.result()
                    datasets[dataset_id] = dataset
                    bar_loading_data.update()
        else:
            for dataset_dir in dataset_dirs:
                dataset_id, dataset = loadDataset_local(dataset_dir)
                datasets[dataset_id] = dataset
                bar_loading_data.update()
    except KeyboardInterrupt:
        manager.stop()
        executor._processes.clear()
        os.kill(os.getpid(), 9)
    bar_loading_data.close()

    results = {"time": {}, "ram": {}}
    result_measures = []

    bar_running_analysis = manager.counter(
        total=len(methods), desc="Running Analysis", unit="methods"
    )
    logging.info(" > Starting Complexity analysis")
    for method in methods:
        method_name = get_method_name_from_docker_image(method)
        logging.info(f" - {method_name}")
        complexity_benchmark_result_folder = os.path.join(
            out_folder, "runs", method_name
        )
        shutil.rmtree(complexity_benchmark_result_folder, ignore_errors=True)
        client = docker.from_env()
        logging.info(" > Pulling Image")
        try:
            image = client.images.get(method)
        except docker.errors.ImageNotFound:
            logging.info("Image does not exist. Pulling it...")
            client.images.pull(method)

        summed_results = {"time": [], "ram": []}
        for r in range(n_repeats):
            failing = False
            for i, n in enumerate(n_samples):
                complexity_benchmark_result_folder_run = os.path.join(
                    complexity_benchmark_result_folder, str(r)
                )
                runner = DockerMethodRunner(
                    method,
                    datasets[n],
                    "evaluation",
                    hyperparameters[method_name],
                    resultsFolder=complexity_benchmark_result_folder_run,
                    debug=additionalOutput,
                    capture_docker_stats=True,
                )
                result = runner.run(cpu_count=1, mem_limit="20g")
                if result is None:
                    if failing:
                        # if we failed twice in a row, we assume that the method is not working
                        logging.error(
                            f"Failed to run {method_name} on dataset {n} repeatedly. Skipping further attempts!"
                        )
                        break
                    failing = True

            parallel = True
            result_folders = glob(
                os.path.join(complexity_benchmark_result_folder_run, "*")
            )
            run_results = []
            if parallel == True:
                with ProcessPoolExecutor() as executor:
                    # submit all tasks and get future objects
                    futures = [
                        executor.submit(load_result, folder, try_load_docker_stats=True)
                        for folder in result_folders
                    ]
                    # process results from tasks in order of task completion
                    for future in as_completed(futures):
                        result = future.result()
                        run_results.append(result)
            else:
                for experiment_result in result_folders:
                    run_results.append(load_result(experiment_result, True))

            run_results = pd.DataFrame(run_results)
            run_results["number"] = (
                run_results["dataset"].str.split("-").str[2].astype(int)
            )
            for i, n in enumerate(n_samples):
                if not (run_results["number"] == n).any():
                    # If there is no result for this number of samples, we add a an empty result
                    run_results = run_results.append(
                        {
                            "number": n,
                        },
                        ignore_index=True,
                    )

            sorted_results = run_results.sort_values(by=["number"])

            summed_results["time"] = np.append(
                summed_results["time"], sorted_results["method_time"]
            )
            summed_results["ram"] = np.append(
                summed_results["ram"], sorted_results["memory_max"]
            )

        value_matrix_time = summed_results["time"].reshape(
            (len(n_samples), n_repeats), order="F"
        )
        summed = np.sum(value_matrix_time, axis=1)
        scaled = summed / summed.max()
        scaled = np.nan_to_num(scaled, nan=1)
        best_cpu, rest = big_o.infer_big_o_class(
            sorted_results["number"], scaled, simplicity_bias=0.004
        )
        classes = pd.DataFrame({"class": rest.keys(), "residual": rest.values()})
        classes.to_csv(os.path.join(out_folder, f"complexities_time_{method_name}.csv"))

        value_matrix_ram = summed_results["ram"].reshape(
            (len(n_samples), n_repeats), order="F"
        )
        summed = np.sum(value_matrix_ram, axis=1)
        scaled = summed / summed.max()
        scaled = np.nan_to_num(scaled, nan=1)
        best_ram, rest = big_o.infer_big_o_class(
            sorted_results["number"], scaled, simplicity_bias=0.00004
        )
        classes = pd.DataFrame({"class": rest.keys(), "residual": rest.values()})
        classes.to_csv(os.path.join(out_folder, f"complexities_ram_{method_name}.csv"))

        results["time"][method] = best_cpu
        results["ram"][method] = best_ram

        dataseries = {
            f"time_overall_{method_name}": np.average(value_matrix_time, axis=1),
            f"ram_overall_{method_name}": np.average(value_matrix_ram, axis=1),
        }
        for n in range(n_repeats):
            dataseries[f"time_run_{n}_{method_name}"] = value_matrix_time.T[n].tolist()
            dataseries[f"ram_run_{n}_{method_name}"] = value_matrix_ram.T[n].tolist()
        measures = pd.DataFrame(
            dataseries,
            index=sorted_results["number"].to_list(),
        )
        result_measures.append(measures)
        # measures.to_csv(os.path.join(out_folder, "measures_raw.csv"))
        # pd.DataFrame(list(others.items())[1:8]).to_csv(
        #     os.path.join(out_folder, "big_o.csv"), header=False, index=False
        # )
        bar_running_analysis.update()

        # Cooldown for 10 seconds
        time.sleep(10)

    bar_running_analysis.close()
    manager.stop()
    logging.info(f"Exporting results to {out_folder}")
    results = pd.DataFrame(
        {
            "Leakage Detection Method": results["time"].keys(),
            "Time Complexity": results["time"].values(),
            "RAM Complexity": results["ram"].values(),
        }
    )
    results.to_csv(os.path.join(out_folder, "results.csv"), index=False)

    # TODO: Escape complexity formula into math mode
    results.style.hide(axis="index").to_latex(os.path.join(out_folder, "results.tex"))

    result_measures = pd.concat(result_measures, axis=1)
    result_measures.to_csv(os.path.join(out_folder, "measures.csv"))
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Scaled Figure
    overall_measures = result_measures[
        [col for col in result_measures.columns if "overall" in col]
    ]
    plot = (overall_measures / overall_measures.max()).plot()
    plot.set_title(f"Complexity Analysis (scaled): {style}")

    ### Add complexities in background

    x = np.arange(0, n_max, 1)

    values = pd.DataFrame(
        {
            "const": 1,
            "log": np.log(x),
            "linear": x,
            "poly": x**4,
        },
        index=x,
    )

    values["expo"] = x
    values["expo"] = values["expo"].astype(object)
    values["expo"] = 2 ** values["expo"]

    plot = (values["const"] / values["const"].max()).plot(alpha=0.2, color="black")
    (values["log"] / values["log"].max()).plot(alpha=0.2, color="black")
    (values["linear"] / values["linear"].max()).plot(alpha=0.2, color="black")
    (values["poly"] / values["poly"].max()).plot(alpha=0.2, color="black")
    (values["expo"] / values["expo"].max()).plot(alpha=0.2, color="black")

    fig = plot.get_figure()
    fig.savefig(os.path.join(out_folder, "measures.png"))
    plt.close(fig)

    # Raw Time Values
    plot = result_measures[
        [
            col
            for col in result_measures.columns
            if ("time" in col and not "overall" in col)
        ]
    ].plot(alpha=0.2, color="black")
    overall_measures[[col for col in overall_measures.columns if "time" in col]].plot(
        ax=plot
    )
    plot.set_title(f"{style}: Time Values")
    fig = plot.get_figure()
    fig.savefig(os.path.join(out_folder, "time.png"))
    plt.close(fig)

    plot = result_measures[
        [
            col
            for col in result_measures.columns
            if ("ram" in col and not "overall" in col)
        ]
    ].plot(alpha=0.2, color="black")
    overall_measures[[col for col in overall_measures.columns if "ram" in col]].plot(
        ax=plot
    )
    plot.set_title(f"{style}: RAM Values")
    fig = plot.get_figure()
    fig.savefig(os.path.join(out_folder, "ram.png"))
    plt.close(fig)
    return results
