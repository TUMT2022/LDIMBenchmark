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
    out_dir = os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED, "synthetic-days-10")
    generateDatasetForTimeSpanDays(4, out_dir)
    dataset = Dataset(out_dir).loadData()


def test_generator_junction():
    out_dir = os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED, "synthetic-j-4")
    generateDatasetForJunctionNumber(4, out_dir)
    dataset = Dataset(out_dir).loadData()


def test_generator_set_time():
    one_out_dir = os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED, "synthetic-days-2")
    generateDatasetsForTimespan(1, 3, os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED))
    dataset = Dataset(one_out_dir).loadData()


def test_generator_set_junctions():
    one_out_dir = os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED, "synthetic-j-6")
    generateDatasetsForJunctions(
        5, 7, os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED)
    )
    dataset = Dataset(one_out_dir).loadData()
