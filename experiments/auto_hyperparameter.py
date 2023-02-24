from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from ldimbenchmark.benchmark import LDIMBenchmark
from ldimbenchmark.classes import LDIMMethodBase
from typing import Dict, List
from ldimbenchmark.evaluation_metrics import f1Score
from ldimbenchmark.methods import LILA, MNF
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from tqdm import tqdm
import itertools
import logging

logLevel = "INFO"

numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(
    level=numeric_level,
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logging.getLogger().setLevel(numeric_level)


param_grid = {
    "lila": {
        # "est_length": np.arange(1, 100, 8).tolist(),
        # "C_threshold": np.arange(-0.3, 1, 0.1).tolist(),
        # "delta": np.arange(0, 10, 1).tolist(),
        # Best
        "est_length": 169.0,
        "C_threshold": 8.0,
        "delta": 8.0,
    },
    "mnf": {
        "gamma": np.arange(-0.3, 1, 0.05).tolist(),
        "window": [1, 5, 10, 20],
    },
    "dualmethod": {"est_length": 480.0, "C_threshold": 0.4, "delta": 0.4},
}


datasets = DatasetLibrary("test_data/datasets").download(DATASETS.BATTLEDIM)


benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets=datasets,
    results_dir="./grid-search",
    debug=True,
    # multi_parameters=True,
)
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/lila:0.1.20"])

# execute benchmark
# benchmark.run_benchmark(parallel=True, parallel_max_workers=3)

results = benchmark.evaluate(current_only=False)


# def run_benchmark(method: LDIMMethodBase, dataset: Dataset, params: Dict):
#     benchmark = LDIMBenchmark(
#         {method.name: {dataset.id: params}},
#         dataset,
#         results_dir="./grid-search",
#         debug=True,
#     )
#     benchmark.add_local_methods(method)

#     # execute benchmark
#     benchmark.run_benchmark()

#     results = benchmark.evaluate()
#     return results, params


# class GridSearch:
#     """
#     Search for best hyperparameter fit for a Leakage Detection Method.
#     The cartesian product of all inputs in the param_grid will be tested with the method used.
#     """

#     def __init__(self, method: LDIMMethodBase, param_grid: Dict[str, List]) -> None:
#         self.method = method
#         self.param_grid = param_grid

#     def detect_offline(self, dataset: Dataset, parallel=True):
#         """
#         Runs the method.
#         """
#         # Create Matrix with all Combinations
#         index = pd.MultiIndex.from_product(
#             self.param_grid.values(), names=self.param_grid.keys()
#         )
#         param_matrix = pd.DataFrame(index=index).reset_index()
#         # print(param_matrix)

#         self.results = {}
#         preliminary_results = []

#         arguments_list = zip(
#             itertools.repeat(self.method, param_matrix.shape[0]),
#             itertools.repeat(dataset, param_matrix.shape[0]),
#             [param_matrix.iloc[n].to_dict() for n in range(param_matrix.shape[0])],
#         )

#         if parallel:
#             with Pool(processes=cpu_count() - 1) as p:
#                 jobs = [
#                     p.apply_async(func=run_benchmark, args=arguments)
#                     for arguments in arguments_list
#                 ]
#                 for job in tqdm(jobs):
#                     job_result, params = job.get()  # Execute Job
#                     job_result["params"] = [params]
#                     print(job_result)
#                     preliminary_results.append(job_result)
#         else:
#             for arguments in tqdm(arguments_list):
#                 job_result, params = run_benchmark(*arguments)
#                 job_result["params"] = params
#                 preliminary_results.append(job_result)

#         results = pd.concat(preliminary_results)
#         results = results.sort_values(by=["F1"])
#         results.to_csv("results.csv")

#     def detect_online(self, dataset: Dataset):
#         pass


# datasets = DatasetLibrary("tests/test_data/datasets").download(DATASETS.BATTLEDIM)


# # param_grid = {
# #     "gamma": np.arange(-0.3, 1, 0.05).tolist(),
# #     "window": [1, 5, 10, 20],
# # }
# # grid_search = GridSearch(MNF(), param_grid)

# param_grid = {
#     "est_length": np.arange(1, 100, 8).tolist(),
#     "C_threshold": np.arange(-0.3, 1, 0.05).tolist(),
#     "delta": np.arange(0, 10, 0.5).tolist(),
# }
# param_grid = {
#     "est_length": [1, 72],
#     "C_threshold": [3.0],
#     "delta": [4.0],
# }
# grid_search = GridSearch(LILA(), param_grid)


# grid_search.detect_offline(datasets[0].loadData().loadBenchmarkData())

# print(grid_search.results)

# from sklearn import svm, datasets
# from sklearn.model_selection import GridSearchCV

# iris = datasets.load_iris()
# clf = GridSearchCV(ScikitLearnEstimatorAdapter(), param_grid)
# clf.fit([{}], [{}], cv=1)
# print(clf.cv_results_)
