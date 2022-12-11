import os

from ldimbenchmark.datasets import Dataset

import numpy as np
import scipy.stats as stats

from typing import Literal

from collections.abc import Sequence
from numpy.random import Generator, PCG64


class DatasetDerivator:
    """
    Chaos Monkey for your Dataset.
    It changes the values of the dataset (in contrast to DatasetTransformer, which changes only structure of the dataset)

    Generate Noise, make sensors fail, skew certain dataseries

    Add underlying long term trends

    """

    def __init__(self, datasets: Dataset | list[Dataset], out_path: str):

        # TODO: Check if datasets is a list or a single dataset
        if isinstance(datasets, Sequence):
            self.datasets: list[Dataset] = datasets
        else:
            self.datasets: list[Dataset] = [datasets]
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
                # TODO Implement derivates
                for value in values:
                    loadedDataset = Dataset(dataset.path).loadDataset()

                    data = getattr(loadedDataset, apply_to)
                    noise = self.__get_random_norm(value, data.index.shape)

                    # TODO; move below for derviation
                    data = data.mul(1 + noise, axis=0)

                    setattr(loadedDataset, apply_to, data)

                    loadedDataset.info["derivations"] = {}
                    loadedDataset.info["derivations"]["data"] = []
                    loadedDataset.info["derivations"]["data"].append(
                        {
                            "to": apply_to,
                            "kind": derivation,
                            "value": value,
                        }
                    )
                    loadedDataset._update_id()
                    derivedDatasetPath = os.path.join(
                        self.out_path, loadedDataset.id + "/"
                    )

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