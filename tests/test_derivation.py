from ldimbenchmark.datasets.classes import (
    Dataset,
    DatasetInfo,
    DatasetInfoDatasetProperty,
    DatasetInfoDatasetObject,
)
from ldimbenchmark.generator.poulakis_network import generatePoulakisNetwork
from ldimbenchmark.datasets.derivation import DatasetDerivator
from tests.shared import TEST_DATA_FOLDER_DATASETS_BATTLEDIM, TEST_DATA_FOLDER_DATASETS

from unittest.mock import Mock
import pytest
import tempfile
import yaml
import pandas as pd
import numpy as np
from wntr.network import write_inpfile, to_dict
from pandas.testing import assert_frame_equal
import os


def test_derivator_model(snapshot, mocked_dataset1: Dataset):
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_model(
        "junctions", "elevation", "accuracy", [0.1]
    )
    snapshot.assert_match(to_dict(derivedDatasets[0].loadDataset().model))


# def test_derivator_data():
#     dataset = Dataset(TEST_DATA_FOLDER_BATTLEDIM)

#     derivator = DatasetDerivator([dataset], TEST_DATA_FOLDER_DATASETS)

#     derivator.derive_data("demands", "noise", [0.1, 0.2, 0.3])


def test_derivator_data_demands(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: demands (and no others)"""
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_data("demands", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadDataset().demands.to_csv())
    assert_frame_equal(
        mocked_dataset1.loadDataset().flows, derivedDatasets[0].loadDataset().flows
    )
    assert_frame_equal(
        mocked_dataset1.loadDataset().levels, derivedDatasets[0].loadDataset().levels
    )
    assert_frame_equal(
        mocked_dataset1.loadDataset().pressures,
        derivedDatasets[0].loadDataset().pressures,
    )


def test_derivator_data_pressures(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: pressures (and no others)"""
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_data("pressures", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadDataset().pressures.to_csv())
    assert_frame_equal(
        mocked_dataset1.loadDataset().flows, derivedDatasets[0].loadDataset().flows
    )
    assert_frame_equal(
        mocked_dataset1.loadDataset().levels, derivedDatasets[0].loadDataset().levels
    )
    assert_frame_equal(
        mocked_dataset1.loadDataset().demands,
        derivedDatasets[0].loadDataset().demands,
    )


def test_derivator_data_flows(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: flows (and no others)"""
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_data("flows", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadDataset().flows.to_csv())
    assert_frame_equal(
        mocked_dataset1.loadDataset().demands, derivedDatasets[0].loadDataset().demands
    )
    assert_frame_equal(
        mocked_dataset1.loadDataset().levels, derivedDatasets[0].loadDataset().levels
    )
    assert_frame_equal(
        mocked_dataset1.loadDataset().pressures,
        derivedDatasets[0].loadDataset().pressures,
    )


def test_derivator_data_levels(snapshot, mocked_dataset1: Dataset):
    """Testing Derivation for data: levels (and no others)"""
    derivator = DatasetDerivator([mocked_dataset1], TEST_DATA_FOLDER_DATASETS)
    derivedDatasets = derivator.derive_data("levels", "precision", [0.1])
    snapshot.assert_match(derivedDatasets[0].loadDataset().levels.to_csv())
    assert_frame_equal(
        mocked_dataset1.loadDataset().flows, derivedDatasets[0].loadDataset().flows
    )
    assert_frame_equal(
        mocked_dataset1.loadDataset().demands, derivedDatasets[0].loadDataset().demands
    )
    assert_frame_equal(
        mocked_dataset1.loadDataset().pressures,
        derivedDatasets[0].loadDataset().pressures,
    )


def test_derivator_data_sampling(snapshot, mocked_dataset_time: Dataset):
    """Testing Derivation for data: levels (and no others)"""
    derivator = DatasetDerivator(
        [mocked_dataset_time], TEST_DATA_FOLDER_DATASETS, force=True
    )
    derivedDatasets = derivator.derive_data("levels", "downsample", [540])
    snapshot.assert_match(derivedDatasets[0].loadDataset().levels.to_csv())
    assert_frame_equal(
        mocked_dataset_time.loadDataset().flows, derivedDatasets[0].loadDataset().flows
    )
    assert_frame_equal(
        mocked_dataset_time.loadDataset().demands,
        derivedDatasets[0].loadDataset().demands,
    )
    assert_frame_equal(
        mocked_dataset_time.loadDataset().pressures,
        derivedDatasets[0].loadDataset().pressures,
    )
