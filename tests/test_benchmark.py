import os
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


def test_hyperparameters_base_configurations():
    # Root level
    hyperparameters = {
        "param1": 1,
        "param2": "test",
    }
    matching_parameters = LDIMBenchmark._get_hyperparameters_for_methods_and_datasets(
        dataset_base_ids=["dataset1"],
        method_ids=["method1"],
        hyperparameters=hyperparameters,
    )
    assert matching_parameters == {
        "method1": {
            "dataset1": hyperparameters,
        },
    }

    # Method Specific
    hyperparameters = {
        "method1": {
            "param1": 1,
            "param2": "test",
        },
        "method2": {
            "param3": "test2",
        },
    }
    matching_parameters = LDIMBenchmark._get_hyperparameters_for_methods_and_datasets(
        dataset_base_ids=["dataset1"],
        method_ids=["method1", "method2"],
        hyperparameters=hyperparameters,
    )
    assert matching_parameters == {
        "method1": {
            "dataset1": hyperparameters["method1"],
        },
        "method2": {
            "dataset1": hyperparameters["method2"],
        },
    }

    # Method and Database Specific
    hyperparameters = {
        "method1": {
            "dataset1": {
                "param1": 1,
                "param2": "test",
            }
        }
    }
    matching_parameters = LDIMBenchmark._get_hyperparameters_for_methods_and_datasets(
        dataset_base_ids=["dataset1"],
        method_ids=["method1"],
        hyperparameters=hyperparameters,
    )
    assert matching_parameters == {
        "method1": {
            "dataset1": hyperparameters["method1"]["dataset1"],
        },
    }

    # Method and Database Specific
    hyperparameters = {
        "method1": {
            "dataset1": {
                "param1": 1,
                "param2": "test",
            }
        },
        "method2": {
            "dataset1": {
                "param1": 1,
                "param2": "test",
            },
            "dataset-name": {
                "param3": 1,
                "param4": "test",
            },
        },
    }
    matching_parameters = LDIMBenchmark._get_hyperparameters_for_methods_and_datasets(
        dataset_base_ids=["dataset1", "dataset-name"],
        method_ids=["method1", "method2"],
        hyperparameters=hyperparameters,
    )
    assert matching_parameters == {
        "method1": {
            "dataset1": hyperparameters["method1"]["dataset1"],
            "dataset-name": {},
        },
        "method2": {
            "dataset1": hyperparameters["method2"]["dataset1"],
            "dataset-name": hyperparameters["method2"]["dataset-name"],
        },
    }

    # Method and Database (derivates) Specific
    hyperparameters = {
        "method1": {
            "dataset1": {
                "param1": 1,
                "param2": "test",
            }
        }
    }
    matching_parameters = LDIMBenchmark._get_hyperparameters_for_methods_and_datasets(
        dataset_base_ids=["dataset1", "dataset1-23409823490"],
        method_ids=["method1"],
        hyperparameters=hyperparameters,
    )
    assert matching_parameters == {
        "method1": {
            "dataset1": hyperparameters["method1"]["dataset1"],
            "dataset1-23409823490": hyperparameters["method1"]["dataset1"],
        },
    }

    # Method and Database (only derivates) Specific
    hyperparameters = {
        "method1": {
            "dataset1": {
                "param1": 1,
                "param2": "test",
            }
        }
    }
    matching_parameters = LDIMBenchmark._get_hyperparameters_for_methods_and_datasets(
        dataset_base_ids=["dataset1-23409823490"],
        method_ids=["method1"],
        hyperparameters=hyperparameters,
    )
    assert matching_parameters == {
        "method1": {
            "dataset1-23409823490": hyperparameters["method1"]["dataset1"],
        },
    }


def test_get_hyperparameter_matrix():
    multi_hyperparameters = {
        "param1": [1, 2, 3],
        "param2": ["test1", "test2"],
    }

    test = LDIMBenchmark._get_hyperparameters_matrix_from_hyperparameters_with_list(
        multi_hyperparameters
    )
    assert test == [
        {"param1": 1, "param2": "test1"},
        {"param1": 1, "param2": "test2"},
        {"param1": 2, "param2": "test1"},
        {"param1": 2, "param2": "test2"},
        {"param1": 3, "param2": "test1"},
        {"param1": 3, "param2": "test2"},
    ]
