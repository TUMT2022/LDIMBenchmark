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
    generateDatasetForTimeSpanDays(
        90, os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED)
    )


def test_generator_junction():
    generateDatasetForJunctionNumber(
        30, os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED)
    )


def test_generator_set_time():
    generateDatasetsForTimespan(
        1, 10, os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED)
    )


def test_generator_set_junctions():
    generateDatasetsForJunctions(
        4, 10, os.path.join(TEST_DATA_FOLDER_DATASETS_GENERATED)
    )
