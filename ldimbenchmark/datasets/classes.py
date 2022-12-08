import numpy as np
import pandas as pd
import os
import yaml
from wntr.network.io import read_inpfile, write_inpfile
from wntr.network import WaterNetworkModel
from datetime import datetime
from ldimbenchmark.classes import BenchmarkData
import numpy as np
import pandas as pd
import os
from glob import glob
import json
from typing import Literal, TypedDict
from ldimbenchmark.constants import LDIM_BENCHMARK_CACHE_DIR
import shutil
import hashlib
from pandas import DataFrame


class DatasetInfoDatasetOverwrites(TypedDict):
    """
    Dataset Config.yml representation
    """

    file_path: str
    index_column: str
    decimal: str
    delimiter: str


class DatasetInfoDatasetObject(TypedDict):
    """
    Dataset Config.yml representation
    """

    start: datetime
    end: datetime


class DatasetInfoDatasetProperty(TypedDict):
    """
    Dataset Config.yml representation
    """

    evaluation: DatasetInfoDatasetObject
    training: DatasetInfoDatasetObject
    overwrites: DatasetInfoDatasetOverwrites


class DatasetInfo(TypedDict):
    """
    Dataset Config.yml representation
    """

    name: str
    dataset: DatasetInfoDatasetProperty
    inp_file: str


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
            self.info: DatasetInfo = yaml.safe_load(f)

        self.name = self.info["name"]

    def loadBenchmarkData(self):
        return LoadedDataset(self)


class _LoadedDatasetPart:
    """
    A sub-dataset of a loaded dataset (e.g. training or evaluation)
    """

    def __init__(self, dict: dict[str, DataFrame]):
        self.pressures: DataFrame = dict["pressures"]
        self.demands: DataFrame = dict["demands"]
        self.flows: DataFrame = dict["flows"]
        self.levels: DataFrame = dict["levels"]
        self.leaks: DataFrame = dict["leaks"]


class LoadedDataset(Dataset, _LoadedDatasetPart):
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

        super(Dataset, self).__init__(
            dict=DatasetTransformer._loadDatasetsDirectly(
                dataset.path, self.info["dataset"]["overwrites"]
            )
        )

        # TODO: Cache dataset
        # TODO: Run checks as to confirm that the dataset_info.yaml information are right
        # eg. check start and end times

        self.model: WaterNetworkModel = read_inpfile(
            os.path.join(self.path, self.info["inp_file"])
        )

    def loadBenchmarkData(self):
        return BenchmarkDatasets(self)

    def exportTo(self, folder: str):
        """
        Exports the dataset to a given folder
        """
        write_inpfile(self.model, os.path.join(folder, self.info["inp_file"]))
        # TODO: Speed up by multiprocessing
        self.pressures.to_csv(os.path.join(folder, "pressures.csv"))
        self.demands.to_csv(os.path.join(folder, "demands.csv"))
        self.flows.to_csv(os.path.join(folder, "flows.csv"))
        self.levels.to_csv(os.path.join(folder, "levels.csv"))
        with open(os.path.join(folder, f"dataset_info.yaml"), "w") as f:
            yaml.dump(self.info, f)


class BenchmarkDatasets(LoadedDataset):
    """
    A dataset that contains the benchmark data for the training and evaluation dataset
    """

    def __init__(self, dataset: LoadedDataset):
        """
        Loads the benchmark data for the training and evaluation dataset
        """
        (training_dataset, evaluation_dataset) = DatasetTransformer(
            dataset, dataset.info
        ).splitIntoTrainingEvaluationDatasets()
        self.name = dataset.name
        self.model = dataset.model

        # Load Data
        self.train = _LoadedDatasetPart(training_dataset)
        self.evaluation = _LoadedDatasetPart(evaluation_dataset)

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


class DatasetTransformer:
    """
    Transform a dataset to a test dataset

    It does not change the values of the dataset (in contrast to DatasetDerivator, which adds noise)
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
        # TODO: Speed up by multiprocessing
        specific_dataset_dir = os.path.join(self.dataset_dir, type)
        os.makedirs(specific_dataset_dir, exist_ok=True)

        full_dataset = DatasetTransformer._loadDatasetsDirectly(
            self.dataset.path, self.config["dataset"]["overwrites"]
        )

        pressures = full_dataset["pressures"].loc[
            self.config["dataset"][type]["start"] : self.config["dataset"][type]["end"]
        ]
        pressures.to_csv(os.path.join(specific_dataset_dir, "pressures.csv"))
        demands = full_dataset["demands"].loc[
            self.config["dataset"][type]["start"] : self.config["dataset"][type]["end"]
        ]
        demands.to_csv(os.path.join(specific_dataset_dir, "demands.csv"))
        flows = full_dataset["flows"].loc[
            self.config["dataset"][type]["start"] : self.config["dataset"][type]["end"]
        ]
        flows.to_csv(os.path.join(specific_dataset_dir, "flows.csv"))
        levels = full_dataset["levels"].loc[
            self.config["dataset"][type]["start"] : self.config["dataset"][type]["end"]
        ]
        levels.to_csv(os.path.join(specific_dataset_dir, "levels.csv"))

        mask = (
            full_dataset["leaks"]["leak_time_start"]
            > self.config["dataset"][type]["start"]
        ) & (
            full_dataset["leaks"]["leak_time_start"]
            < self.config["dataset"][type]["end"]
        )
        leaks = full_dataset["leaks"][mask]
        levels.to_csv(os.path.join(specific_dataset_dir, "leaks.csv"))

        return {
            "pressures": pressures,
            "demands": demands,
            "flows": flows,
            "levels": levels,
            "leaks": leaks,
        }

    @staticmethod
    def _loadDatasetsDirectly(
        datastet_dir: str, overwrites: DatasetInfoDatasetOverwrites = {}
    ):
        """
        Load the dataset directly from the files
        """
        index_column = "Timestamp"
        delimiter = ","
        decimal = "."

        config_name = "index_column"
        if config_name in overwrites:
            index_column = overwrites[config_name]
        config_name = "delimiter"
        if config_name in overwrites:
            delimiter = overwrites[config_name]
        config_name = "decimal"
        if config_name in overwrites:
            decimal = overwrites[config_name]

        dataset = {}
        for file in filter(
            lambda x: not "leaks" in os.path.basename(x),
            glob(os.path.join(datastet_dir + "/" + "*.csv")),
        ):
            dataset[os.path.basename(file).lower()[:-4]] = pd.read_csv(
                file,
                index_col=index_column,
                parse_dates=True,
                delimiter=delimiter,
                decimal=decimal,
            )
        dataset["leaks"] = pd.read_csv(
            os.path.join(datastet_dir, "leaks.csv"),
            parse_dates=True,
            delimiter=delimiter,
            decimal=decimal,
        )
        return dataset

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
                    training_data = DatasetTransformer._loadDatasetsDirectly(
                        os.path.join(self.dataset_dir, "training"),
                    )
                    evaluation_data = DatasetTransformer._loadDatasetsDirectly(
                        os.path.join(self.dataset_dir, "evaluation")
                    )
                return (training_data, evaluation_data)

        # is not equal, remove old dataset
        shutil.rmtree(self.dataset_dir, ignore_errors=True)
        training_data = self._extractDataType("training")
        evaluation_data = self._extractDataType("evaluation")

        # Create hash file
        with open(self.dataset_hash_file, "w") as f:
            f.write(str(dataset_config_hash))

        return (training_data, evaluation_data)
