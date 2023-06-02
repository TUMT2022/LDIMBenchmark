from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from ldimbenchmark.benchmark import LDIMBenchmark
from ldimbenchmark.classes import LDIMMethodBase
from typing import Dict, List
from ldimbenchmark.evaluation import f1Score
from ldimbenchmark.methods import LILA, MNF, DUALMethod
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from tqdm import tqdm
import itertools
import logging
from multiprocessing import get_context
from multiprocessing_logging import install_mp_handler

import multiprocessing as mp
import queue


def run_benchmark(args):
    method, dataset, params = args
    logLevel = "INFO"

    numeric_level = getattr(logging, logLevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % logLevel)

    logging.basicConfig(
        level=numeric_level,
        handlers=[logging.StreamHandler()],
        format="%(asctime)s | %(processName)s %(levelname)-8s %(message)s",
    )
    logging.getLogger().setLevel(numeric_level)

    benchmark = LDIMBenchmark(
        {method.name: {dataset.id: params}},
        dataset,
        results_dir="./grid-search",
        debug=False,
    )
    benchmark.add_local_methods(method)

    # execute benchmark
    benchmark.run_benchmark()

    results = benchmark.evaluate(write_results=False)
    logging.info("Benchmark finished.")

    return results, params


class GridSearch:
    """
    Search for best hyperparameter fit for a Leakage Detection Method.
    The cartesian product of all inputs in the param_grid will be tested with the method used.
    """

    def __init__(self, method: LDIMMethodBase, param_grid: Dict[str, List]) -> None:
        self.method = method
        self.param_grid = param_grid

    def detect_offline(self, dataset: Dataset, parallel=True):
        """
        Runs the method.
        """
        # Create Matrix with all Combinations
        index = pd.MultiIndex.from_product(
            self.param_grid.values(), names=self.param_grid.keys()
        )
        param_matrix = pd.DataFrame(index=index).reset_index()
        # print(param_matrix)

        self.results = {}
        preliminary_results = []

        arguments_list = list(
            zip(
                itertools.repeat(self.method, param_matrix.shape[0]),
                itertools.repeat(dataset, param_matrix.shape[0]),
                [param_matrix.iloc[n].to_dict() for n in range(param_matrix.shape[0])],
            )
        )

        if parallel:
            with get_context("spawn").Pool(processes=int(cpu_count() - 1)) as p:
                max_ = len(arguments_list)
                with tqdm(total=max_) as pbar:
                    for job_result, params in p.imap_unordered(
                        run_benchmark, arguments_list, chunksize=1
                    ):
                        # job_result, params = a[0]
                        job_result["params"] = [params]
                        # print(job_result)
                        preliminary_results.append(job_result)
                        pbar.update()
                p.close()
                p.join()

            # manager = mp.Manager()
            # stop_event = manager.Event()
            # q = manager.Queue()
            # d = manager.dict()
            # d["region"] = set()
            # pool = mp.get_context("spawn").Pool(mp.cpu_count() + 2)
            # watcher = pool.apply_async(listener, (q, d, stop_event))
            # stop_event.set()
            # jobs = []
            # pbar = tqdm(total=len(arguments_list))

            # for arguments in arguments_list:
            #     job = pool.apply_async(run_benchmark, (q, d, arguments))
            #     jobs.append(job)
            # try:
            #     for job in jobs:
            #         job.get(
            #             300
            #         )  # get the result or throws a timeout exception after 300 seconds
            #         pbar.update()
            # except mp.TimeoutError:
            #     pool.terminate()

            # stop_event.set()  # stop listener process
            # print("process complete")

            # pool = get_context("spawn").Pool(processes=cpu_count() - 1)
            # # pool = Pool(processes=cpu_count() - 1)

            # pbar = tqdm(total=len(arguments_list))

            # def update(*a):
            #     pbar.update()
            #     job_result, params = a[0]
            #     job_result["params"] = [params]
            #     # print(job_result)
            #     preliminary_results.append(job_result)
            #     # tqdm.write(str(a))

            # def updateError(*a):
            #     pbar.update()
            #     tqdm.write(str(a))

            # for arguments in arguments_list:
            #     pool.apply_async(
            #         run_benchmark,
            #         args=arguments,
            #         callback=update,
            #         error_callback=updateError,
            #     )
            # # tqdm.write('scheduled')
            # pool.close()
            # pool.join()

            #
            # with Pool(processes=cpu_count() - 1) as p:
            #     jobs = [
            #         p.apply_async(func=run_benchmark, args=arguments)
            #         for arguments in arguments_list
            #     ]
            #     for job in tqdm(jobs):
            #         job_result, params = job.get()  # Execute Job
            #         logging.info("Got result from job...")
            #         job_result["params"] = [params]
            #         # print(job_result)
            #         preliminary_results.append(job_result)
        else:
            for arguments in tqdm(arguments_list):
                job_result, params = run_benchmark(*arguments)
                job_result["params"] = params
                preliminary_results.append(job_result)

        results = pd.concat(preliminary_results)
        results = results.sort_values(by=["F1"])
        results.to_csv("results.csv")

    def detect_online(self, dataset: Dataset):
        pass


if __name__ == "__main__":
    logLevel = "INFO"

    numeric_level = getattr(logging, logLevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % logLevel)

    logging.basicConfig(
        level=numeric_level,
        handlers=[logging.StreamHandler()],
        format="%(asctime)s | %(processName)s %(levelname)-8s %(message)s",
    )
    logging.getLogger().setLevel(numeric_level)
    # install_mp_handler()

    datasets = DatasetLibrary("tests/test_data/datasets").download(DATASETS.BATTLEDIM)

    # MNF
    # param_grid = {
    #     "gamma": np.arange(-0.3, 1, 0.05).tolist(),
    #     "window": [1, 5, 10, 20],
    # }
    # grid_search = GridSearch(MNF(), param_grid)

    # LILA
    # param_grid = {
    #     "est_length": np.arange(1, 24 * 9, 24).tolist(),
    #     "C_threshold": np.arange(0.0, 10.0, 2).tolist(),
    #     "delta": np.arange(0.0, 10.0, 2).tolist(),
    # }
    # grid_search = GridSearch(LILA(), param_grid)

    # DualModel
    param_grid = {
        "est_length": np.arange(1, 24 * 9, 24).tolist(),
        "C_threshold": np.arange(0.0, 10.0, 2).tolist(),
        "delta": np.arange(0.0, 10.0, 2).tolist(),
    }
    param_grid = {
        "est_length": [24 * 20, 24 * 10],
        "C_threshold": [0.2, 0.8, 1.0],
        "delta": [0.2, 0.8],
    }
    grid_search = GridSearch(DUALMethod(), param_grid)

    # Test
    # param_grid = {
    #     "est_length": [1, 72],
    #     "C_threshold": [3.0],
    #     "delta": [4.0],
    # }
    # grid_search = GridSearch(DUALMethod(), param_grid)

    grid_search.detect_offline(datasets[0].loadDataset().loadBenchmarkData())

    print(grid_search.results)

    # from sklearn import svm, datasets
    # from sklearn.model_selection import GridSearchCV

    # iris = datasets.load_iris()
    # clf = GridSearchCV(ScikitLearnEstimatorAdapter(), param_grid)
    # clf.fit([{}], [{}], cv=1)
    # print(clf.cv_results_)
