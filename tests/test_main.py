import os
import yaml
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    LocalMethodRunner,
    DockerMethodRunner,
    FileBasedMethodRunner,
)
from ldimbenchmark.methods.dualmethod import DUALMethod
from tests.method_to_test import YourCustomLDIMMethod
from ldimbenchmark.methods import LILA, MNF
import pandas as pd
from tests.shared import (
    TEST_DATA_FOLDER,
)
import logging
from pandas.testing import assert_frame_equal


def test_benchmark(mocked_dataset1: Dataset):
    # dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)

    local_methods = [MNF(), LILA(), DUALMethod()]

    hyperparameters = {
        "mnf": {
            "window": 10,
            "gamma": 0.1,
            "_dma_specific": {
                "dma_a": {
                    "window": "10 days",
                    "gamma": 0.1,
                },
            },
        },
        "lila": {"default_flow_sensor": "J-02"},
    }

    benchmark = LDIMBenchmark(
        hyperparameters,
        mocked_dataset1,
        results_dir="./benchmark-results",
        debug=True,
    )
    benchmark.add_local_methods(local_methods)

    # .add_docker_methods(methods)

    # execute benchmark
    benchmark.run_benchmark(
        evaluation_mode="training"
        # parallel=True,
    )

    benchmark.evaluate()
    # benchmark.evaluate(
    #     generate_plots=True,
    # )


# def test_complexity():
#     local_methods = [YourCustomLDIMMethod()]  # , LILA()]

#     hyperparameter = {}

#     benchmark = LDIMBenchmark(hyperparameter, [], results_dir="./benchmark-results")
#     benchmark.add_local_methods(local_methods)

#     # .add_docker_methods(methods)

#     # execute benchmark
#     benchmark.run_complexity_analysis(
#         methods=local_methods,
#         style="time",
#         # parallel=True,
#     )

#     # benchmark.evaluate()


def test_single_run_local(mocked_dataset1: Dataset):
    runner = LocalMethodRunner(
        detection_method=YourCustomLDIMMethod(),
        dataset=mocked_dataset1,
        hyperparameters={},
        resultsFolder="./benchmark-results/runner_results",
    )
    runner.run()

    pass


# def test_single_run_docker(mocked_dataset1: Dataset):
#     results_folder = "./benchmark-results/runner_results"
#     runner = DockerMethodRunner(
#         "testmethod",
#         mocked_dataset1,
#         resultsFolder=results_folder,
#     )
#     # TODO: Refactor to read result from file
#     result = runner.run()
#     asserted_results = pd.DataFrame(YourCustomLDIMMethod.get_results())
#     detected_leaks = pd.read_csv(
#         os.path.join(runner.resultsFolder, "detected_leaks.csv")
#     )
#     assert_frame_equal(asserted_results, detected_leaks)


def test_method(mocked_dataset1: Dataset):
    trainData = (
        mocked_dataset1.loadData().loadBenchmarkData().getTrainingBenchmarkData()
    )
    evaluationData = (
        mocked_dataset1.loadData().loadBenchmarkData().getEvaluationBenchmarkData()
    )

    method = YourCustomLDIMMethod()
    method.prepare(trainData)
    method.detect_offline(evaluationData)
    pass


def test_method_file_based(mocked_dataset1: Dataset):
    args_dir = os.path.join(TEST_DATA_FOLDER, "args")
    out_dir = os.path.join(TEST_DATA_FOLDER, "out")

    os.makedirs(args_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(args_dir, "options.yml"), "w") as f:
        yaml.dump(
            {
                "dataset_part": "evaluation",
                "hyperparameters": {},
                "goal": "detection",
                "stage": "detect",
                "method": "offline",
                "debug": False,
            },
            f,
        )
    runner = FileBasedMethodRunner(
        detection_method=YourCustomLDIMMethod(),
        inputFolder=mocked_dataset1.path,
        argumentsFolder=args_dir,
        outputFolder=out_dir,
    )
    runner.run()
