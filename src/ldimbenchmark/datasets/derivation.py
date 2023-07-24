from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
import itertools
import logging
import math
import os
import random
import shutil
import tempfile
import enlighten

from pandas import DataFrame
import pandas as pd
from ldimbenchmark.constants import CPU_COUNT

from ldimbenchmark.datasets import Dataset

import numpy as np
import scipy.stats as stats

from typing import Literal, Union, List

from collections.abc import Sequence
from numpy.random import Generator, PCG64


def get_random_norm(
    noise_level: float, size: int, seed: int = 27565124760782368551060429849508057759
):
    """
    Generate a random normal distribution with a given noise level
    """
    random_gen = Generator(PCG64(seed))
    lower, upper = -noise_level, noise_level
    mu, sigma = 0, noise_level / 3
    # truncnorm_gen =
    # truncnorm_gen.random_state =
    X = stats.truncnorm(
        (lower - mu) / sigma,
        (upper - mu) / sigma,
        loc=mu,
        scale=sigma,
    )
    return X.rvs(
        size,
        random_state=random_gen,
    )


def get_random_selection(
    size: int, selection: int, seed: int = 27565124760782368551060429849508057759
):
    """
    Generate a random normal distribution with a given noise level
    """
    random.seed(seed)
    return random.sample(range(size), selection)


def _apply_derivation_to_DataFrame(
    derivation: Literal["precision", "sensitivity", "downsample"],
    value: float,
    dataframe: DataFrame,
    key: str,
) -> DataFrame:
    if derivation == "precision":
        if value >= 1:
            raise Exception("Precision value must be smaller than 1")
        noise = get_random_norm(value, dataframe.index.shape)
        dataframe = dataframe.mul(1 + noise, axis=0)
    elif derivation == "sensitivity":
        if value["shift"] == "top":
            dataframe = np.ceil(dataframe / value["value"]) * value["value"]
        if value["shift"] == "middle":
            dataframe = np.floor(dataframe / value["value"]) * value["value"] + (
                value["value"] / 2
            )
        else:
            dataframe = np.floor(dataframe / value["value"]) * value["value"]

    elif derivation == "downsample":
        dataframe = dataframe.reset_index()
        dataframe = dataframe.groupby(
            (dataframe["Timestamp"] - dataframe["Timestamp"][0]).dt.total_seconds()
            // (value),
            group_keys=True,
        ).first()
        dataframe = dataframe.set_index("Timestamp")

    elif derivation == "count":
        pass
    else:
        raise ValueError(f"Derivation {derivation} not implemented")

    return (key, dataframe)


def try_load_derivation_datasets(folder):
    try:
        dataset = Dataset(folder)
        return (
            dataset.id,
            str(dataset.info["derivations"]) if "derivations" in dataset.info else None,
        )
    except Exception:
        return None


class DatasetDerivator:
    """
    Chaos Monkey for your Dataset.
    It changes the values of the dataset (in contrast to DatasetTransformer, which changes only structure of the dataset)

    Generate Noise, make sensors fail, skew certain dataseries

    Add underlying long term trends

    """

    def __init__(
        self,
        datasets: Union[Dataset, List[Dataset]],
        out_path: str,
        ignore_cache: bool = False,
    ):
        if not isinstance(datasets, Sequence):
            datasets = [datasets]
        self.datasets: List[Dataset] = datasets
        self.out_path = out_path

        if not ignore_cache:
            derivation_folders = glob(os.path.join(out_path, "*"))
            dataset_derivations = []
            parallel = True
            if parallel == True:
                with ProcessPoolExecutor() as executor:
                    # submit all tasks and get future objects
                    futures = [
                        executor.submit(try_load_derivation_datasets, folder)
                        for folder in derivation_folders
                    ]
                    # process results from tasks in order of task completion
                    for future in as_completed(futures):
                        derivation = future.result()
                        if derivation is not None:
                            dataset_derivations.append(derivation)
            else:
                for folder in derivation_folders:
                    derivation = try_load_derivation_datasets(folder)
                    if derivation is not None:
                        dataset_derivations.append(derivation)

            self.cached_derivations = DataFrame(dataset_derivations)
        else:
            self.cached_derivations = DataFrame(columns=[0, 1])

        self.ignore_cache = ignore_cache

        self.all_derived_datasets = []

    def get_dervived_datasets(self, with_original: bool = False):
        if with_original:
            return self.datasets + self.all_derived_datasets
        return self.all_derived_datasets

    # TODO: Parallelization

    def derive_model(
        self,
        # TODO: Add Pattern derivation
        apply_to: Literal["junctions", "pipes"],  # , "patterns"],
        change_property: Literal["elevation", "diameter", "length", "roughness"],
        derivation: Literal["accuracy"],
        values: list,
    ):
        """
        Derives a new dataset from the original one.

        :param derivation: Name of derivation that should be applied
        :param values: Values for the derivation
        """

        newDatasets = []
        for dataset in self.datasets:
            for value in values:
                this_dataset = Dataset(dataset.path)
                this_dataset.info["derivations"] = {}
                this_dataset.info["derivations"]["model"] = []
                this_dataset.info["derivations"]["model"].append(
                    {
                        "element": apply_to,
                        "property": change_property,
                        "value": value,
                    }
                )
                this_dataset._update_id()

                temp_dir = tempfile.TemporaryDirectory()
                temporaryDatasetPath = temp_dir.name

                cache_entry = self.cached_derivations[
                    (
                        self.cached_derivations[1]
                        == str(this_dataset.info["derivations"])
                    )
                    & self.cached_derivations[0].str.contains(this_dataset.name)
                ]
                if len(cache_entry) > 1:
                    logging.warning(
                        f"more than one cache entry found: {str(cache_entry[0].values)}"
                    )
                if len(cache_entry) < 1:
                    logging.info(
                        f"Generating Model Derivation for {this_dataset.id} with derivations {str(this_dataset.info['derivations']['model'])}"
                    )
                    loadedDataset = this_dataset.loadData()

                    # Derive
                    if (
                        derivation == "accuracy"
                        and change_property == "elevation"
                        and apply_to == "junctions"
                    ):
                        junctions = loadedDataset.model.junction_name_list
                        noise = get_random_norm(value, len(junctions))
                        for index, junction in enumerate(junctions):
                            model_node = loadedDataset.model.get_node(junction)
                            if model_node.length + noise[index] > 0:
                                model_node.elevation += noise[index]
                            else:
                                logging.warning(
                                    f"Junction {model_node.name} would have a negative elevation. Skipping alteration."
                                )

                    elif (
                        derivation == "accuracy"
                        and change_property == "roughness"
                        and apply_to == "pipes"
                    ):
                        pipes = loadedDataset.model.pipe_name_list
                        noise = get_random_norm(value, len(pipes))
                        for index, pipe in enumerate(pipes):
                            model_pipe = loadedDataset.model.get_link(pipe)
                            if model_pipe.length + noise[index] > 0:
                                model_pipe.roughness += noise[index]
                            else:
                                logging.warning(
                                    f"Pipe {model_pipe.name} would have a negative roughness. Skipping alteration."
                                )
                    elif (
                        derivation == "accuracy"
                        and change_property == "diameter"
                        and apply_to == "pipes"
                    ):
                        pipes = loadedDataset.model.pipe_name_list
                        noise = get_random_norm(value, len(pipes))
                        for index, pipe in enumerate(pipes):
                            model_pipe = loadedDataset.model.get_link(pipe)
                            if model_pipe.length + noise[index] > 0:
                                model_pipe.diameter += noise[index]
                            else:
                                logging.warning(
                                    f"Pipe {model_pipe.name} would have a negative diameter. Skipping alteration."
                                )
                    elif (
                        derivation == "accuracy"
                        and change_property == "length"
                        and apply_to == "pipes"
                    ):
                        pipes = loadedDataset.model.pipe_name_list
                        noise = get_random_norm(value, len(pipes))
                        for index, pipe in enumerate(pipes):
                            # TODO: If length is to short, we would get negative values so we need to skip the alteration
                            model_pipe = loadedDataset.model.get_link(pipe)
                            if model_pipe.length + noise[index] > 0:
                                model_pipe.length += noise[index]
                            else:
                                logging.warning(
                                    f"Pipe {model_pipe.name} would have a negative length. Skipping alteration."
                                )

                    else:
                        raise Exception(
                            f"No derivation '{derivation}' for {apply_to} {change_property} implemented"
                        )

                    # TODO: Make sanity checks for data:
                    # - No negative values

                    # Save
                    # os.makedirs(os.path.dirname(temporaryDatasetPath), exist_ok=True)
                    loadedDataset.exportTo(temporaryDatasetPath)

                    tmp_dataset = Dataset(temporaryDatasetPath)
                    logging.info(f"Saved {loadedDataset.id}")
                    logging.info("Populating cache")
                    tmp_dataset.is_valid()
                    tmp_dataset.loadData()
                    dataset_path = os.path.join(self.out_path, tmp_dataset.id + "/")
                    shutil.copytree(
                        temporaryDatasetPath, dataset_path, dirs_exist_ok=True
                    )

                    new_dataset = Dataset(dataset_path)

                    newDatasets.append(new_dataset)
                    self.all_derived_datasets.append(new_dataset)

                    temp_dir.cleanup()
                else:
                    # Dataset already generated
                    cached_dataset = Dataset(
                        os.path.join(self.out_path, cache_entry.iloc[0][0])
                    )
                    newDatasets.append(cached_dataset)
                    self.all_derived_datasets.append(cached_dataset)

        return newDatasets

    def derive_data(
        self,
        apply_to: Union[
            Literal["demands", "levels", "pressures", "flows"],
            List[Literal["demands", "levels", "pressures", "flows"]],
        ],
        # TODO: Add Chaos Monkey, introducing missing values, skewed values (way out of bound),
        # TODO: Add simple skew (static, or linear)
        derivation: Literal["sensitivity", "precision", "downsample", "count"],
        options_list: Union[List[dict], List[float]],
    ):
        """
        Derives a new dataset from the original one.

        :param derivation: Name of derivation that should be applied
        :param options_list: List of options for the derivation

        ``derivation="precision"``
            Adds noise to the data. The noise is normally distributed with a mean of 0 and a standard deviation of ``value``.

        ``derivation="sensitivity"``
            Simulates a sensor with a certain sensitivity. Meaning data will be rounded to the nearest multiple of ``value``.
            ``shift`` determines how the dataseries is shifted. ``"top"`` shifts the dataseries to the top, ``"bottom"`` to the bottom and ``"middle"`` to the middle.
            Default for shift is "bottom"
            e.g.
            realvalues = [1.1, 1.2, 1.3, 1.4, 1.5] and ``value=0.5`` and ``shift="top"`` will result in [1.5, 1.5, 1.5, 1.5, 2]
            realvalues = [1.1, 1.2, 1.3, 1.4, 1.5] and ``value=0.5`` and ``shift="bottom"`` will result in [1, 1, 1, 1, 1.5]

        ``derivation="downsample"``
            Simulates a sensor with less readings per timeframe.
            Values must be given in seconds.

        ``derivation="count"``
            Simulates less number of sensors. The sensors are chosen randomly (but deterministic).
            ``value`` determines the percentage of sensors in relation to the pipe count.


        """

        if not isinstance(apply_to, list):
            apply_to = [apply_to]

        newDatasets = []
        for dataset in self.datasets:
            for options in options_list:
                abort = False
                # Prepare data for derivation
                this_dataset = Dataset(dataset.path)
                if "derivations" not in this_dataset.info:
                    this_dataset.info["derivations"] = {}
                if "data" not in this_dataset.info["derivations"]:
                    this_dataset.info["derivations"]["data"] = []

                # Apply derivation
                value = options
                if derivation == "precision" or derivation == "downsample":
                    if isinstance(value, dict):
                        value = value["value"]

                if derivation == "sensitivity":
                    if not isinstance(value, dict):
                        value = {
                            "value": value,
                            "shift": "middle",
                        }

                    shift = value["value"]
                    if value["shift"] == "bottom":
                        shift = 0
                    if value["shift"] == "middle":
                        shift = value["value"] / 2
                    # value["value"] = shift

                # Save Derivation
                for application in apply_to:
                    this_dataset.info["derivations"]["data"].append(
                        {
                            "to": application,
                            "kind": derivation,
                            "value": value,
                        }
                    )
                cache_entry = self.cached_derivations[
                    (
                        self.cached_derivations[1]
                        == str(this_dataset.info["derivations"])
                    )
                    & self.cached_derivations[0].str.contains(this_dataset.name)
                ]
                if len(cache_entry) > 1:
                    raise Exception(
                        f"more than one cache entry found: {str(cache_entry[0].values)}"
                    )
                if len(cache_entry) < 1:
                    temp_dir = tempfile.TemporaryDirectory()
                    temporaryDatasetPath = temp_dir.name
                    logging.info(
                        f"Generating Data Derivation for {this_dataset.id} with derivations {str(this_dataset.info['derivations']['data'])}"
                    )
                    loadedDataset = this_dataset.loadData()

                    manager = enlighten.get_manager()
                    for application in apply_to:
                        datasets = getattr(loadedDataset, application)
                        keys = list(datasets.keys())

                        if derivation == "count":
                            # Value is supplied as percentile, we need to convert it
                            value = value / 100
                            # For count derivation we do not need to make changes fo the data, just remove some sensors
                            current_sensor_ratio = len(keys) / len(
                                this_dataset.model.pipe_name_list
                            )
                            logging.debug(
                                f"Current Sensor Ratio: {current_sensor_ratio}"
                            )
                            if value > current_sensor_ratio:
                                logging.warning(
                                    f"Value {value} is larger than the current sensor ratio {current_sensor_ratio}. Aborting derivation."
                                )
                                abort = True
                                break

                            mask = get_random_selection(
                                len(keys),
                                len(keys)
                                - round(len(keys) * (value / current_sensor_ratio)),
                            )
                            logging.debug(f"Removed {len(mask)} sensors")
                            for i in mask:
                                del datasets[keys[i]]
                            keys = list(datasets.keys())

                        else:
                            # For all other derivations we need to apply the derivation to the data
                            transformations = zip(
                                itertools.repeat(derivation, len(keys)),
                                itertools.repeat(value, len(keys)),
                                [datasets[key] for key in keys],
                                keys,
                            )
                            bar_derivations = manager.counter(
                                total=len(keys),
                                desc=f"Deriving {application}",
                                unit="sensors",
                            )
                            bar_derivations.refresh()

                            # logging.debug(filepaths)
                            with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                                # submit all tasks and get future objects
                                futures = [
                                    executor.submit(
                                        _apply_derivation_to_DataFrame,
                                        derivation,
                                        value,
                                        dataset_key,
                                        key,
                                    )
                                    for derivation, value, dataset_key, key in transformations
                                ]
                                # process results from tasks in order of task completion
                                for future in as_completed(futures):
                                    key, result = future.result()
                                    datasets[key] = result
                                    if len(result) <= 3:
                                        logging.warn(
                                            "Derived data would only have three data points. That's not a proper dataset anymore. Aborting."
                                        )

                                        abort = True

                                    bar_derivations.update()
                            bar_derivations.close()
                        setattr(loadedDataset, application, datasets)

                    if not abort:
                        logging.info(f"Exporting dataset")
                        os.makedirs(
                            os.path.dirname(temporaryDatasetPath), exist_ok=True
                        )
                        loadedDataset.exportTo(temporaryDatasetPath)

                        tmp_dataset = Dataset(temporaryDatasetPath)
                        logging.info(f"Saved {loadedDataset.id}")
                        logging.info("Populating cache")
                        tmp_dataset.is_valid()
                        tmp_dataset.loadData()
                        dataset_path = os.path.join(self.out_path, tmp_dataset.id + "/")
                        shutil.copytree(
                            temporaryDatasetPath, dataset_path, dirs_exist_ok=True
                        )

                        new_dataset = Dataset(dataset_path)

                        newDatasets.append(new_dataset)
                        self.all_derived_datasets.append(new_dataset)

                    temp_dir.cleanup()
                    manager.stop()

                else:
                    # Dataset already generated
                    cached_dataset = Dataset(
                        os.path.join(self.out_path, cache_entry.iloc[0][0])
                    )
                    newDatasets.append(cached_dataset)
                    self.all_derived_datasets.append(cached_dataset)

        return newDatasets
