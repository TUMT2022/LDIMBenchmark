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
)
import tempfile
import yaml
import pandas as pd
import numpy as np
from wntr.network import write_inpfile
import os


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def mocked_dataset():
    os.makedirs(TEST_DATA_FOLDER_DATASETS_TEST, exist_ok=True)
    temp_dir = tempfile.TemporaryDirectory(dir=TEST_DATA_FOLDER_DATASETS_TEST)
    with open(temp_dir.name + "/dataset_info.yaml", "w") as f:
        f.write(
            yaml.dump(
                DatasetInfo(
                    name="test",
                    inp_file="model.inp",
                    dataset=DatasetInfoDatasetProperty(
                        evaluation=DatasetInfoDatasetObject(
                            start="2018-01-01 00:10:00",
                            end="2018-01-01 00:19:00",
                        ),
                        training=DatasetInfoDatasetObject(
                            start="2018-01-01 00:00:00",
                            end="2018-01-1 00:09:00",
                        ),
                    ),
                )
            )
        )
    # Datapoints
    for dataset in ["demands", "levels", "flows", "pressures"]:
        pd.DataFrame(
            {
                "a": np.ones(20),
            },
            index=pd.date_range(
                start="2018-01-01 00:00:00", end="2018-01-01 00:19:00", freq="T"
            ),
        ).to_csv(temp_dir.name + "/" + dataset + ".csv", index_label="Timestamp")

    # Leaks
    pd.DataFrame(
        {
            "leak_pipe_id": "test",
            "leak_pipe_nodes": "['A', 'B']",
            "leak_diameter": 0.1,
            "leak_area": 0.1,
            "leak_time_start": "2018-01-01 00:01:00",
            "leak_time_peak": "2018-01-01 00:03:00",
            "leak_time_end": "2018-01-01 00:10:00",
            "leak_max_flow": 0.1,
        },
        index=[0],
    ).to_csv(temp_dir.name + "/leaks.csv")

    write_inpfile(generatePoulakisNetwork(), temp_dir.name + "/model.inp")
    yield Dataset(temp_dir.name)
    # temp_dir.cleanup()
