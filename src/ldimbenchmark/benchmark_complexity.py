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
from matplotlib.ticker import FormatStrFormatter
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

from ldimbenchmark.utilities import convert_byte_size, get_method_name_from_docker_image


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

    if n_max < 1:
        raise ValueError("n_max must be at least 1")
    if n_measures < 1:
        raise ValueError("n_measures must be at least 1")
    logging.info("Complexity Analysis:")
    logging.info(" > Generating Datasets")
    if style == "periods":
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

    results = {"time": {}, "ram": {}, "time_avg": {}, "ram_avg": {}}
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
                    cpu_count=1,
                    mem_limit="20g",
                )
                result = runner.run()
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

        results["time"][method_name] = best_cpu
        results["ram"][method_name] = best_ram
        results["time_avg"][method_name] = np.average(
            value_matrix_time[~np.isnan(value_matrix_time)],
        )
        results["ram_avg"][method_name] = np.average(
            value_matrix_ram[~np.isnan(value_matrix_ram)]
        )

        dataseries = {
            f"time_overall_{method_name}": np.average(value_matrix_time, axis=1),
            f"memory_overall_{method_name}": np.average(value_matrix_ram, axis=1),
        }
        for n in range(n_repeats):
            # Use underscores to hide hide the labels in the plot
            dataseries[f"_time_run_{n}_{method_name}"] = value_matrix_time.T[n].tolist()
            dataseries[f"_memory_run_{n}_{method_name}"] = value_matrix_ram.T[
                n
            ].tolist()
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
            "Method": results["time"].keys(),
            "time": results["time"].values(),
            "time_avg": results["time_avg"].values(),
            "ram": results["ram"].values(),
            "ram_avg": results["ram_avg"].values(),
        }
    )
    results["time"] = results["time"].map(lambda x: str(x).split(":")[0].lower())
    results["ram"] = results["ram"].map(lambda x: str(x).split(":")[0].lower())
    results["time_avg"] = results["time_avg"].map(
        lambda x: "\SI{" + "{:.2}".format(x) + "}{\second}"
    )
    results["ram_avg"] = results["ram_avg"].map(
        lambda x: "\SI{" + "{:,.0f}".format(x / 1024 / 1024) + "}{\mega\\byte}"
    )
    results = results.set_index("Method")

    results.columns = pd.MultiIndex.from_arrays(
        [
            [
                "Time",
                "Time",
                "Memory",
                "Memory",
            ],
            ["complexity", "average", "complexity", "average"],
        ]
    )

    results.to_csv(os.path.join(out_folder, "results.csv"))

    results.style.set_table_styles(
        [
            # {'selector': 'toprule', 'props': ':hline;'},
            {"selector": "midrule", "props": ":hline;"},
            # {'selector': 'bottomrule', 'props': ':hline;'},
        ],
        overwrite=False,
    ).to_latex(
        os.path.join(out_folder, "results.tex"),
        label=f"table:complexity:{style}",
        caption=f"Complexities of the different methods depending on the amount of {style}.",
        column_format="l|" + str("l" * (len(results.columns))),
        position="H",
        multicol_align="l",
        position_float="centering",
    )

    result_measures = pd.concat(result_measures, axis=1)
    result_measures.to_csv(os.path.join(out_folder, "measures.csv"))
    mpl.rcParams.update(mpl.rcParamsDefault)

    # Scaled Figure
    overall_measures = result_measures[
        [col for col in result_measures.columns if "overall" in col]
    ]
    overall_measures_labels = [
        col.replace("_overall", "")
        for col in result_measures.columns
        if "overall" in col
    ]
    ax = (overall_measures / overall_measures.max()).plot()
    # ax.set_title(f"Complexity Analysis for different {style} inputs")

    ### Add complexities in background

    x = np.arange(0, n_max, 1)

    values = pd.DataFrame(
        {
            "_const": 1,
            "_log": np.log(x),
            "_linear": x,
            "_poly": x**4,
        },
        index=x,
    )

    values["_expo"] = x
    values["_expo"] = values["_expo"].astype(object)
    values["_expo"] = 2 ** values["_expo"]

    ax = (values["_const"] / values["_const"].max()).plot(alpha=0.2, color="black")
    (values["_log"] / values["_log"].max()).plot(alpha=0.2, color="black")
    (values["_linear"] / values["_linear"].max()).plot(alpha=0.2, color="black")
    (values["_poly"] / values["_poly"].max()).plot(alpha=0.2, color="black")
    (values["_expo"] / values["_expo"].max()).plot(alpha=0.2, color="black")

    if style == "junctions":
        ax.set_xlabel("junction number")
    elif style == "periods":
        ax.set_xlabel("time period [d]")

    ax.set_ylabel("scale")
    ax.legend(overall_measures_labels, loc="lower right")
    fig = ax.get_figure()

    fig.savefig(
        os.path.join(out_folder, "measures.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    # Raw Time Values
    ax = result_measures[
        [
            col
            for col in result_measures.columns
            if ("time" in col and not "overall" in col)
        ]
    ].plot(alpha=0.2, color="black", label="_nolegend")
    for col in overall_measures.columns:
        if "time" in col:
            overall_measures[col].plot(
                ax=ax,
                label=col.replace("_overall", ""),
            )
    # ax.set_title(f"Complexity Analysis for different {style} inputs")
    ax.set_xlabel("time period [d]")
    ax.set_ylabel("time [s]")
    ax.legend(loc="lower right")

    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(out_folder, "time.png"),
        bbox_inches="tight",
    )
    plt.close(fig)

    ax = result_measures[
        [
            col
            for col in result_measures.columns
            if ("memory" in col and not "overall" in col)
        ]
    ].plot(alpha=0.2, color="black", label="_nolegend")
    for col in overall_measures.columns:
        if "memory" in col:
            overall_measures[col].plot(
                ax=ax,
                label=col.replace("_overall", ""),
            )
    # ax.set_title(f"Complexity Analysis for different {style} inputs")
    ax.set_xlabel("junction number")
    ax.set_ylabel("memory [B]")
    ax.legend(loc="lower right")
    # convert labels to byte units
    ax.set_yticklabels([convert_byte_size(x) for x in ax.get_yticks()])

    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    fig = ax.get_figure()
    fig.savefig(
        os.path.join(out_folder, "memory.png"),
        bbox_inches="tight",
    )
    plt.close(fig)
    return results
