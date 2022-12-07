"""
dataset_library.py
"""


from enum import Enum
import tempfile
from ldimbenchmark.datasets.load_battledim import BattledimDatasetLoader

import numpy as np
import pandas as pd
import os
import yaml
from wntr.network.io import read_inpfile, write_inpfile
from wntr.network import WaterNetworkModel
from glob import glob
from datetime import datetime
import json
from typing import Literal, TypedDict
from ldimbenchmark.constants import LDIM_BENCHMARK_CACHE_DIR

from ldimbenchmark.classes import BenchmarkData, DatasetInfo
import shutil
import hashlib
import numpy as np

# class syntax
class DATASETS(Enum):
    """
    Enum of available datasets
    """

    BATTLEDIM = "battledim"


class Dataset:
    """
    The lighweight dataset class (no data loaded)
    """

    def __init__(self, path):
        """
        :param path: Path to the dataset (or where to download it)
        """

        self.path = path
        # Read dataset_info.yaml
        with open(os.path.join(path, "dataset_info.yaml")) as f:
            self.info = yaml.safe_load(f)

        self.name = self.info["name"]

    def load(self):
        return LoadedDataset(self)


class _LoadedDatasetPart:
    """
    A sub-dataset of a loaded dataset (e.g. training or evaluation)
    """

    def __init__(self, dict: dict):
        self.pressures: np.DataFrame = dict["pressures"]
        self.demands = dict["demands"]
        self.flows = dict["flows"]
        self.levels = dict["levels"]


class LoadedDataset(Dataset):
    """
    The heavy dataset class (data loaded)
    Represents the Low Level Interface as Code + Some Convience Methods
    """

    def __init__(self, dataset: Dataset):
        """
        Loads a dataset from a given path.
        Config values Example:
            dataset:
                evaluation:
                    start: 2019-01-01 00:00
                    end: 2019-12-31 23:55
                    overwrites:
                    filePath: "2019"
                    index_column: Timestamp
                    delimiter: ;
                    decimal: ','
                training:
                    start: 2018-01-01 00:00
                    end: 2019-12-31 23:55
                    overwrites:
                    filePath: "2018"
                    index_column: Timestamp
                    delimiter: ;
                    decimal: ','
        """
        # Keep already loaded data
        self.path = dataset.path
        self.name = dataset.name
        self.info = dataset.info

        self.pressures = pd.DataFrame()
        self.demands = pd.DataFrame()
        self.flows = pd.DataFrame()
        self.levels = pd.DataFrame()

        # TODO: Cache dataset
        # TODO load full dataset here and only split into training and evaluation on demand
        (training_dataset, evaluation_dataset) = DatasetTransformer(
            dataset, dataset.info
        ).splitIntoTrainingEvaluationDatasets()
        # Load Data
        self.train = _LoadedDatasetPart(training_dataset)
        self.evaluation = _LoadedDatasetPart(evaluation_dataset)

        # Load Leaks
        # TODO: make leaks load from .csv
        train_dataset_leakages = []
        evaluation_dataset_leakages = []
        for leak in self.info["leakages"]:
            startTime = datetime.fromisoformat(leak["leak_start_time"])
            leak = {
                "pipe_id": leak["leak_pipe"],
                "leak_start": datetime.fromisoformat(leak["leak_start_time"]),
                "leak_end": datetime.fromisoformat(leak["leak_end_time"]),
                "leak_peak": datetime.fromisoformat(leak["leak_peak_time"]),
                "leak_area": leak["leak_area"] if "leak_area" in leak else 0,
                "leak_diameter": leak["leak_diameter"]
                if "leak_diameter" in leak
                else 0,
                "leak_max_flow": leak["leak_max_flow"]
                if "leak_max_flow" in leak
                else 0,
            }

            if (
                startTime > self.train.pressures.index[0]
                and startTime < self.train.pressures.index[-1]
            ):
                train_dataset_leakages.append(leak)

            if (
                startTime > self.evaluation.pressures.index[0]
                and startTime < self.evaluation.pressures.index[-1]
            ):
                evaluation_dataset_leakages.append(leak)

        self.leaks_evaluation = evaluation_dataset_leakages
        self.leaks_train = train_dataset_leakages

        # TODO: Run checks as to confirm that the dataset_info.yaml information are right
        # eg. check start and end times

        self.model: WaterNetworkModel = read_inpfile(
            os.path.join(self.path, self.info["inp_file"])
        )

    def getTrainingBenchmarkData(self):
        return BenchmarkData(
            pressures=self.train.pressures,
            demands=self.train.demands,
            flows=self.train.flows,
            levels=self.train.levels,
            model=self.model,
        )

    def getEvaluationBenchmarkData(self):
        return BenchmarkData(
            pressures=self.evaluation.pressures,
            demands=self.evaluation.demands,
            flows=self.evaluation.flows,
            levels=self.evaluation.levels,
            model=self.model,
        )

    def exportTo(self, folder: str):
        """
        Exports the dataset to a given folder
        """
        write_inpfile(self.model, os.path.join(folder, self.info["inp_file"]))
        self.pressures.to_csv(os.path.join(folder, "pressures.csv"))
        self.demands.to_csv(os.path.join(folder, "demands.csv"))
        self.flows.to_csv(os.path.join(folder, "flows.csv"))
        self.levels.to_csv(os.path.join(folder, "levels.csv"))
        with open(os.path.join(folder, f"dataset_info.yaml"), "w") as f:
            yaml.dump(self.info, f)


class DatasetLibrary:
    """
    Library of datasets
    """

    def __init__(self, download_path: str):
        self.path = download_path

    def download(self, dataset_name: str | list[str], force=False):
        """
        Downloads a dataset from the internet

        :param dataset_name: Name of the dataset to download
        :param force: Force download even if dataset is already downloaded
        """
        # TODO: Try to import download module for "dataset_name" and execute it.
        # TODO: Implement force download

        if dataset_name is str:
            dataset_name = [DATASETS[dataset_name.upper()]]
        if isinstance(dataset_name, DATASETS):
            dataset_name = [dataset_name]

        datasets = []

        for dataset in dataset_name:
            print("Downloading dataset: " + dataset.value)
            tempdir = tempfile.TemporaryDirectory()
            dataset_download_path = os.path.join(self.path, dataset.value)
            if dataset == DATASETS.BATTLEDIM:
                BattledimDatasetLoader.downloadBattledimDataset(tempdir.name)
                BattledimDatasetLoader.prepareBattledimDataset(
                    tempdir.name, dataset_download_path
                )

                datasets.append(Dataset(dataset_download_path))

            tempdir.cleanup()


class DatasetTransformer:
    """
    Transform a dataset to a test dataset
    """

    def __init__(
        self,
        dataset: Dataset,
        config: DatasetInfo,
        cache_dir: str = LDIM_BENCHMARK_CACHE_DIR,
    ):
        self.dataset = dataset
        self.config = config
        self.cache_dir = cache_dir

        self.dataset_dir = os.path.join(self.cache_dir, config["name"])
        self.dataset_hash_file = os.path.join(self.dataset_dir, ".hash")

    def _extractDataType(
        self,
        type: Literal["training", "evaluation"],
    ):
        if type != "training" and type != "evaluation":
            raise ValueError("type must be either 'training' or 'evaluation'")

        specific_dataset_dir = os.path.join(self.dataset_dir, type)
        os.makedirs(specific_dataset_dir, exist_ok=True)

        index_column = "Timestamp"
        delimiter = ","
        decimal = "."
        base_path = os.path.join(self.dataset.path)

        overwrites_key = "overwrites"
        if overwrites_key in self.config["dataset"][type]:
            config_name = "filePath"
            if config_name in self.config["dataset"][type][overwrites_key]:
                base_path = (
                    os.path.join(
                        base_path,
                        self.config["dataset"][type][overwrites_key][config_name],
                    )
                    + "/"
                )

            config_name = "index_column"
            if config_name in self.config["dataset"][type][overwrites_key]:
                index_column = self.config["dataset"][type][overwrites_key][config_name]
            config_name = "delimiter"
            if config_name in self.config["dataset"][type][overwrites_key]:
                delimiter = self.config["dataset"][type][overwrites_key][config_name]
            config_name = "decimal"
            if config_name in self.config["dataset"][type][overwrites_key]:
                decimal = self.config["dataset"][type][overwrites_key][config_name]

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        dataset = {}
        for file in glob(base_path + "*.csv"):
            dataset[os.path.basename(file).lower()[:-4]] = pd.read_csv(
                file,
                index_col="Timestamp",
                parse_dates=True,
                delimiter=delimiter,
                decimal=decimal,
            )

        pressures = dataset["pressures"].loc[
            self.config["dataset"][type]["start"] : self.config["dataset"][type]["end"]
        ]
        pressures.to_csv(os.path.join(specific_dataset_dir, "pressures.csv"))
        demands = dataset["demands"].loc[
            self.config["dataset"][type]["start"] : self.config["dataset"][type]["end"]
        ]
        demands.to_csv(os.path.join(specific_dataset_dir, "demands.csv"))
        flows = dataset["flows"].loc[
            self.config["dataset"][type]["start"] : self.config["dataset"][type]["end"]
        ]
        flows.to_csv(os.path.join(specific_dataset_dir, "flows.csv"))
        levels = dataset["levels"].loc[
            self.config["dataset"][type]["start"] : self.config["dataset"][type]["end"]
        ]
        levels.to_csv(os.path.join(specific_dataset_dir, "levels.csv"))

        return {
            "pressures": pressures,
            "demands": demands,
            "flows": flows,
            "levels": levels,
        }

    def loadDatasetsDirectly(self, type: str):
        """
        Load the dataset directly from the files
        """
        specific_dataset_dir = os.path.join(self.dataset_dir, type)
        pressures = pd.read_csv(
            os.path.join(specific_dataset_dir, "pressures.csv"),
            index_col="Timestamp",
            parse_dates=True,
        )
        demands = pd.read_csv(
            os.path.join(specific_dataset_dir, "demands.csv"),
            index_col="Timestamp",
            parse_dates=True,
        )
        flows = pd.read_csv(
            os.path.join(specific_dataset_dir, "flows.csv"),
            index_col="Timestamp",
            parse_dates=True,
        )
        levels = pd.read_csv(
            os.path.join(specific_dataset_dir, "levels.csv"),
            index_col="Timestamp",
            parse_dates=True,
        )

        return {
            "pressures": pressures,
            "demands": demands,
            "flows": flows,
            "levels": levels,
        }

    def splitIntoTrainingEvaluationDatasets(
        self,
    ):
        """
        Splits the dataset into training and evaluation datasets, according to the configration in dataset_info.yml

        """

        dataset_config = self.config["dataset"]
        # Hash of configuration params
        # TODO: also include the files themselves
        dataset_config_hash = hashlib.md5(
            json.dumps(dataset_config, sort_keys=True).encode("utf-8")
        ).hexdigest()

        # Check if dataset is already transformed
        if os.path.exists(self.dataset_dir) and os.path.exists(self.dataset_hash_file):
            with open(self.dataset_hash_file, "r") as f:
                existing_hash = f.read()
                if existing_hash == dataset_config_hash:
                    # is equal, no need to transform, just load
                    training_data = self.loadDatasetsDirectly("training")
                    evaluation_data = self.loadDatasetsDirectly("evaluation")
                return (training_data, evaluation_data)

        # is not equal, remove old dataset
        shutil.rmtree(self.dataset_dir, ignore_errors=True)
        training_data = self._extractDataType("training")
        evaluation_data = self._extractDataType("evaluation")

        # Create hash file
        with open(self.dataset_hash_file, "w") as f:
            f.write(str(dataset_config_hash))

        return (training_data, evaluation_data)
