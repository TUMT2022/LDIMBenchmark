import pytest
from ldimbenchmark.datasets.classes import (
    Dataset,
    DatasetInfo,
    DatasetInfoDatasetProperty,
    DatasetInfoDatasetObject,
)
from ldimbenchmark.generator.poulakis_network import generatePoulakisNetwork
from ldimbenchmark.datasets.derivation import DatasetDerivator

from tests.shared import (
    TEST_DATA_FOLDER_DATASETS_TEST,
    TEST_DATA_FOLDER_DATASETS_TEST2,
    TEST_DATA_FOLDER_DATASETS_TEST_TIME,
)
import tempfile
import yaml
import pandas as pd
import numpy as np
from wntr.network import write_inpfile
import os
import logging


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture(autouse=True)
def change_log_level(request):
    logLevel = "INFO"

    numeric_level = getattr(logging, logLevel, None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % logLevel)

    logging.basicConfig(level=numeric_level, handlers=[logging.StreamHandler()])
    logging.getLogger().setLevel(numeric_level)


mocked_dataset_nodes = ["J-02", "J-03"]


@pytest.fixture
def mocked_dataset1():
    os.makedirs(TEST_DATA_FOLDER_DATASETS_TEST, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir=TEST_DATA_FOLDER_DATASETS_TEST)
    dataset_path = temp_dir.name
    # For debugging
    # dataset_path = TEST_DATA_FOLDER_DATASETS_TEST
    with open(os.path.join(dataset_path, "dataset_info.yaml"), "w") as f:
        f.write(
            yaml.dump(
                DatasetInfo(
                    name="test1",
                    inp_file="model.inp",
                    dataset=DatasetInfoDatasetProperty(
                        training=DatasetInfoDatasetObject(
                            start="2018-01-01 00:00:00",
                            end="2018-01-1 00:09:00",
                        ),
                        evaluation=DatasetInfoDatasetObject(
                            start="2018-01-01 00:10:00",
                            end="2018-01-01 00:19:00",
                        ),
                    ),
                )
            )
        )
    # Datapoints
    for dataset in ["demands", "levels", "flows", "pressures"]:
        os.makedirs(os.path.join(dataset_path, dataset), exist_ok=True)
        for sensor in mocked_dataset_nodes:
            pd.DataFrame(
                {
                    sensor: np.ones(20),
                },
                index=pd.date_range(
                    start="2018-01-01 00:00:00", end="2018-01-01 00:19:00", freq="T"
                ),
            ).to_csv(
                os.path.join(dataset_path, dataset, f"{sensor}.csv"),
                index_label="Timestamp",
            )

    # Leaks
    pd.DataFrame(
        {
            "leak_pipe_id": "test",
            "leak_pipe_nodes": str(mocked_dataset_nodes),
            "leak_diameter": 0.1,
            "leak_area": 0.1,
            "leak_time_start": "2018-01-01 00:01:00",
            "leak_time_peak": "2018-01-01 00:03:00",
            "leak_time_end": "2018-01-01 00:10:00",
            "leak_max_flow": 0.1,
        },
        index=[0],
    ).to_csv(os.path.join(dataset_path, "leaks.csv"))

    write_inpfile(generatePoulakisNetwork(), os.path.join(dataset_path, "model.inp"))
    yield Dataset(dataset_path)
    temp_dir.cleanup()


@pytest.fixture
def mocked_dataset2():
    os.makedirs(TEST_DATA_FOLDER_DATASETS_TEST2, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir=TEST_DATA_FOLDER_DATASETS_TEST2)
    dataset_path = temp_dir.name
    # For debugging
    # dataset_path = TEST_DATA_FOLDER_DATASETS_TEST
    with open(os.path.join(dataset_path, "dataset_info.yaml"), "w") as f:
        f.write(
            yaml.dump(
                DatasetInfo(
                    name="test2",
                    inp_file="model.inp",
                    dataset=DatasetInfoDatasetProperty(
                        training=DatasetInfoDatasetObject(
                            start="2018-01-01 00:00:00",
                            end="2018-01-1 00:09:00",
                        ),
                        evaluation=DatasetInfoDatasetObject(
                            start="2018-01-01 00:10:00",
                            end="2018-01-01 00:19:00",
                        ),
                    ),
                )
            )
        )
    # Datapoints
    for dataset in ["demands", "levels", "flows", "pressures"]:
        os.makedirs(os.path.join(dataset_path, dataset), exist_ok=True)
        for sensor in mocked_dataset_nodes:
            pd.DataFrame(
                {
                    sensor: np.linspace(0, 7, num=20),
                },
                index=pd.date_range(
                    start="2018-01-01 00:00:00", end="2018-01-01 00:19:00", freq="T"
                ),
            ).to_csv(
                os.path.join(dataset_path, dataset, f"{sensor}.csv"),
                index_label="Timestamp",
            )

    # Leaks
    pd.DataFrame(
        {
            "leak_pipe_id": "test",
            "leak_pipe_nodes": str(mocked_dataset_nodes),
            "leak_diameter": 0.1,
            "leak_area": 0.1,
            "leak_time_start": "2018-01-01 00:01:00",
            "leak_time_peak": "2018-01-01 00:03:00",
            "leak_time_end": "2018-01-01 00:10:00",
            "leak_max_flow": 0.1,
        },
        index=[0],
    ).to_csv(os.path.join(dataset_path, "leaks.csv"))

    write_inpfile(generatePoulakisNetwork(), os.path.join(dataset_path, "model.inp"))
    yield Dataset(dataset_path)
    temp_dir.cleanup()


@pytest.fixture
def mocked_dataset_time():
    os.makedirs(TEST_DATA_FOLDER_DATASETS_TEST_TIME, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir=TEST_DATA_FOLDER_DATASETS_TEST_TIME)
    dataset_path = temp_dir.name
    # For debugging
    # dataset_path = TEST_DATA_FOLDER_DATASETS_TEST_TIME
    with open(os.path.join(dataset_path, "dataset_info.yaml"), "w") as f:
        f.write(
            yaml.dump(
                DatasetInfo(
                    name="test_time",
                    inp_file="model.inp",
                    dataset=DatasetInfoDatasetProperty(
                        training=DatasetInfoDatasetObject(
                            start="2018-01-01 00:00:00",
                            end="2018-01-1 00:09:00",
                        ),
                        evaluation=DatasetInfoDatasetObject(
                            start="2018-01-01 00:10:00",
                            end="2018-01-01 00:19:00",
                        ),
                    ),
                )
            )
        )
    # Datapoints
    for dataset in ["demands", "levels", "flows", "pressures"]:
        os.makedirs(os.path.join(dataset_path, dataset), exist_ok=True)
        for sensor in mocked_dataset_nodes:
            pd.DataFrame(
                {
                    sensor: range(20),
                },
                index=pd.date_range(
                    start="2018-01-01 00:00:00", end="2018-01-01 00:19:00", freq="T"
                ),
            ).to_csv(
                os.path.join(dataset_path, dataset, f"{sensor}.csv"),
                index_label="Timestamp",
            )

    # Leaks
    pd.DataFrame(
        {
            "leak_pipe_id": "test",
            "leak_pipe_nodes": str(mocked_dataset_nodes),
            "leak_diameter": 0.1,
            "leak_area": 0.1,
            "leak_time_start": "2018-01-01 00:01:00",
            "leak_time_peak": "2018-01-01 00:03:00",
            "leak_time_end": "2018-01-01 00:10:00",
            "leak_max_flow": 0.1,
        },
        index=[0],
    ).to_csv(os.path.join(dataset_path, "leaks.csv"))

    write_inpfile(generatePoulakisNetwork(), os.path.join(dataset_path, "model.inp"))
    yield Dataset(dataset_path)
    temp_dir.cleanup()
