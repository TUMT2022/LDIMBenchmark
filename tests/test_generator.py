from ldimbenchmark.datasets.classes import Dataset
from ldimbenchmark.generator import (
    generateDatasetForTimeSpanDays,
    generateDatasetsForTimespan,
    generateDatasetsForJunctions,
    generateDatasetForJunctionNumber,
)
from tests.shared import (
    TEST_DATA_FOLDER_DATASETS_GENERATED,
)
import os


def test_generator_time():
    out_dir = os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED, "synthetic-days-90")
    generateDatasetForTimeSpanDays(90, out_dir)
    dataset = Dataset(out_dir).loadData()


def test_generator_junction():
    out_dir = os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED, "synthetic-j-30")
    generateDatasetForJunctionNumber(30, out_dir)
    dataset = Dataset(out_dir).loadData()


def test_generator_set_time():
    generateDatasetsForTimespan(
        1, 10, os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED)
    )


def test_generator_set_junctions():
    generateDatasetsForJunctions(
        4, 10, os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED)
    )
