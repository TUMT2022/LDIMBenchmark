import os

from ldimbenchmark.datasets import Dataset

import numpy as np
import scipy.stats as stats

from typing import Literal, Union, List

from collections.abc import Sequence
from numpy.random import Generator, PCG64


class DatasetDerivator:
    """
    Chaos Monkey for your Dataset.
    It changes the values of the dataset (in contrast to DatasetTransformer, which changes only structure of the dataset)

    Generate Noise, make sensors fail, skew certain dataseries

    Add underlying long term trends

    """

    def __init__(self, datasets: Union[Dataset, List[Dataset]], out_path: str):

        # TODO: Check if datasets is a list or a single dataset
        if isinstance(datasets, Sequence):
            self.datasets: List[Dataset] = datasets
        else:
            self.datasets: List[Dataset] = [datasets]
        self.out_path = out_path

        # TODO: should we always use the same seed?
        seed = 27565124760782368551060429849508057759
        self.random_gen = Generator(PCG64(seed))

    # TODO: Add more derivations, like junction elevation

    # TODO: Caching
    # TODO: cross product of derivations

    def derive_model(
        self,
        apply_to: Literal["junctions", "patterns"],
        change_property: Literal["elevation"],
        derivation: str,
        values: list,
    ):
        """
        Derives a new dataset from the original one.

        :param derivation: Name of derivation that should be applied
        :param values: Values for the derivation
        """

        newDatasets = []
        for dataset in self.datasets:

            if derivation == "noise":

                for value in values:

                    loadedDataset = Dataset(dataset.path).loadDataset()
                    junctions = loadedDataset.model.junction_name_list
                    noise = self.__get_random_norm(value, len(junctions))
                    for index, junction in enumerate(junctions):
                        loadedDataset.model.get_node(junction).elevation += noise[index]

                    loadedDataset.info["derivations"] = {}
                    loadedDataset.info["derivations"]["model"] = []
                    loadedDataset.info["derivations"]["model"].append(
                        {
                            "element": apply_to,
                            "property": change_property,
                            "value": value,
                        }
                    )
                    loadedDataset._update_id()

                    derivedDatasetPath = os.path.join(
                        self.out_path, loadedDataset.id + "/"
                    )

                    os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)
                    loadedDataset.exportTo(derivedDatasetPath)

                    # TODO write to dataser_info.yml and add keys with derivation properties
                    newDatasets.append(Dataset(derivedDatasetPath))
        return newDatasets

    def derive_data(
        self,
        apply_to: Literal["demands", "levels", "pressures", "flows"],
        derivation: str,
        options_list: Union[List[dict], List[float]],
    ):
        """
        Derives a new dataset from the original one.

        :param derivation: Name of derivation that should be applied
        :param options_list: List of options for the derivation

        ``derivation="noise"``
            Adds noise to the data. The noise is normally distributed with a mean of 0 and a standard deviation of ``value``.

        ``derivation="sensitivity"``
            Simulates a sensor with a certain sensitivity. Meaning data will be rounded to the nearest multiple of ``value``.
            ``shift`` determines how the dataseries is shifted. ``"top"`` shifts the dataseries to the top, ``"bottom"`` to the bottom and ``"middle"`` to the middle.
            e.g.
            realvalues = [1.1, 1.2, 1.3, 1.4, 1.5] and ``value=0.5`` and ``shift="top"`` will result in [1.5, 1.5, 1.5, 1.5, 2]
            realvalues = [1.1, 1.2, 1.3, 1.4, 1.5] and ``value=0.5`` and ``shift="bottom"`` will result in [1, 1, 1, 1, 1.5]
            realvalues = [1.1, 1.2, 1.3, 1.4, 1.5] and ``value=0.5`` and ``shift="middle"`` will result in [1.25, 1.25, 1.25, 1.25, 1.75]

        """

        newDatasets = []
        for dataset in self.datasets:
            for options in options_list:
                # Prepare data for derivation
                loadedDataset = Dataset(dataset.path).loadDataset()
                data = getattr(loadedDataset, apply_to)
                loadedDataset.info["derivations"] = {}
                loadedDataset.info["derivations"]["data"] = []

                # Apply derivation
                # TODO Implement derivates
                if derivation == "noise":

                    value = options
                    if isinstance(value, dict):
                        value = value["value"]
                    noise = self.__get_random_norm(value, data.index.shape)
                    data = data.mul(1 + noise, axis=0)

                if derivation == "sensitivity":

                    shift = options["value"]
                    if options["shift"] == "bottom":
                        shift = 0
                    if options["shift"] == "middle":
                        shift = options["value"] / 2
                    data = np.divmod(data, options["value"])[0] + shift

                # Save Derivation
                loadedDataset.info["derivations"]["data"].append(
                    {
                        "to": apply_to,
                        "kind": derivation,
                        "value": value,
                    }
                )
                setattr(loadedDataset, apply_to, data)
                loadedDataset._update_id()
                derivedDatasetPath = os.path.join(self.out_path, loadedDataset.id + "/")

                os.makedirs(os.path.dirname(derivedDatasetPath), exist_ok=True)
                loadedDataset.exportTo(derivedDatasetPath)

                newDatasets.append(Dataset(derivedDatasetPath))

        return newDatasets

    def _generateNormalDistributedNoise(self, dataset, noiseLevel):
        """
        generate noise in a gaussian way between the low and high level of noiseLevel
        sigma is choosen so that 99.7% of the data is within the noiseLevel bounds

        :param noiseLevel: noise level in percent

        """
        lower, upper = -noiseLevel, noiseLevel
        mu, sigma = 0, noiseLevel / 3
        X = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
        )
        noise = X.rvs(dataset.index.shape)
        return dataset, noise

    def _generateUniformDistributedNoise(self, dataset, noiseLevel):
        """
        generate noise in a uniform way between the low and high level of noiseLevel

        :param noiseLevel: noise level in percent

        """
        noise = np.random.uniform(-noiseLevel, noiseLevel, dataset.index.shape)

        dataset = dataset.mul(1 + noise, axis=0)
        return dataset, noise

    def __get_random_norm(self, noise_level: float, size: int):
        """
        Generate a random normal distribution with a given noise level
        """
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
            random_state=self.random_gen,
        )
