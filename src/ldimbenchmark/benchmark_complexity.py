import importlib
import os
import time
from datetime import datetime
from glob import glob
import logging
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
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib as mpl
from typing import List


def loadDataset_local(dataset_path):
    dataset = Dataset(dataset_path).loadDataset().loadBenchmarkData()
    number = int(os.path.basename(os.path.normpath(dataset_path)).split("-")[-1])
    return (
        number,
        dataset.getTrainingBenchmarkData(),
        dataset.getEvaluationBenchmarkData(),
    )


def run_benchmark_complexity(
    methods: List[LDIMMethodBase],
    cache_dir=os.path.join(LDIM_BENCHMARK_CACHE_DIR, "datagen"),
    out_folder="out/complexity",
    style=None,
    additionalOutput=False,
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
        generateDatasetsForTimespan(1, 91, datasets_dir)
    if style == "junctions":
        datasets_dir = os.path.join(cache_dir, "synthetic-junctions")
        generateDatasetsForJunctions(4, 59, datasets_dir)

    dataset_dirs = glob(datasets_dir + "/*/")

    logging.info(" > Loading Data")
    datasets = {}
    with Pool(processes=cpu_count() - 1) as p:
        max_ = len(dataset_dirs)
        with tqdm(total=max_) as pbar:
            for datasetid, training, evaluation in p.imap_unordered(
                loadDataset_local, dataset_dirs
            ):
                datasets[datasetid] = {}
                datasets[datasetid]["training"] = training
                datasets[datasetid]["evaluation"] = evaluation
                pbar.update()
    # for dataset in dataset_dirs:
    #     (datasetid, training, evaluation) = loadDataset_local(dataset)
    #     datasets[datasetid] = {}
    #     datasets[datasetid]["training"] = training
    #     datasets[datasetid]["evaluation"] = evaluation

    results = {}
    result_measures = []

    # self.experiments.append(
    #                 LocalMethodRunner(
    #                     method,
    #                     dataset,
    #                     self.hyperparameters,
    #                     resultsFolder=self.runner_results_dir,
    #                 )
    #             )

    logging.info(" > Starting Complexity analyis")
    for method in methods:
        logging.info(f" - {method.name}")

        if additionalOutput:
            method.init_with_benchmark_params(
                os.path.join(out_folder, "additional_output_path"), {}
            )

        def runAlgorithmWithNetwork(n):
            method.train(datasets[n]["training"])
            method.detect_offline(datasets[n]["evaluation"])

        # Run Big-O
        best, others = big_o.big_o(
            runAlgorithmWithNetwork,
            big_o.datagen.n_,
            min_n=4,
            max_n=len(dataset_dirs),
            n_measures=10,
            n_repeats=3,
            return_raw_data=True,
        )
        results[method.name] = best

        measures = pd.DataFrame({"times": others["times"]}, index=others["measures"])
        result_measures.append(measures.rename(columns={"times": method.name}))
        measures.to_csv(os.path.join(out_folder, "measures_raw.csv"))
        pd.DataFrame(list(others.items())[1:8]).to_csv(
            os.path.join(out_folder, "big_o.csv"), header=False, index=False
        )

    results = pd.DataFrame(
        {
            "Leakage Detection Method": results.keys(),
            "Time Complexity": results.values(),
        }
    )
    results.to_csv(os.path.join(out_folder, "results.csv"), index=False)

    # TODO: Escape complexity formula into math mode
    results.style.hide(axis="index").to_latex(os.path.join(out_folder, "results.tex"))

    result_measures = pd.concat(result_measures, axis=1)
    result_measures.to_csv(os.path.join(out_folder, "measures.csv"))
    mpl.rcParams.update(mpl.rcParamsDefault)
    plot = (result_measures / result_measures.max()).plot()
    plot.set_title(f"Complexity Analysis: {style}")
    fig = plot.get_figure()
    fig.savefig(os.path.join(out_folder, "measures.png"))
    return results
