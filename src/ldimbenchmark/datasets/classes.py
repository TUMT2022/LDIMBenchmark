from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import pickle
import numpy as np
import pandas as pd
import os
from wntr.network.io import read_inpfile, write_inpfile
from wntr.network import WaterNetworkModel
from datetime import datetime
from ldimbenchmark.classes import BenchmarkData
import numpy as np
import pandas as pd
import os
from glob import glob
import json
from typing import Literal, Optional, TypedDict, Dict
from ldimbenchmark.constants import CPU_COUNT, LDIM_BENCHMARK_CACHE_DIR
import shutil
import hashlib
from pandas import DataFrame
import json
import hashlib
import logging
import copy

import yaml
from yaml import CDumper
from yaml.representer import SafeRepresenter
from ldimbenchmark.utilities import dirhash


# Fix for Timestamp parsing/dumping in yaml
class TSDumper(CDumper):
    pass


def timestamp_representer(dumper, data):
    return SafeRepresenter.represent_datetime(dumper, data.to_pydatetime())


TSDumper.add_representer(datetime, SafeRepresenter.represent_datetime)
TSDumper.add_representer(pd.Timestamp, timestamp_representer)


class DatasetInfoDatasetObject(TypedDict):
    """
    Dataset Config.yml representation
    """

    start: pd.Timestamp
    end: pd.Timestamp


class DatasetInfoDatasetProperty(TypedDict):
    """
    Dataset Config.yml representation
    """

    evaluation: DatasetInfoDatasetObject
    training: DatasetInfoDatasetObject


class DatasetInfoDerivations(TypedDict):
    model: list
    data: list


class DatasetInfo(TypedDict):
    """
    Dataset Config.yml representation
    """

    name: str
    dataset: DatasetInfoDatasetProperty
    inp_file: str
    derivations: DatasetInfoDerivations
    checksum: Optional[str]


class _LoadedDatasetPartNew:
    """
    A sub-dataset of a loaded dataset (e.g. training or evaluation)
    """

    def __init__(self, dict: Dict[str, DataFrame]):
        self.pressures: Dict[str, DataFrame] = dict["pressures"]
        self.demands: Dict[str, DataFrame] = dict["demands"]
        self.flows: Dict[str, DataFrame] = dict["flows"]
        self.levels: Dict[str, DataFrame] = dict["levels"]
        self.leaks: DataFrame = dict["leaks"]


def write_to_file(dataframe, file_path):
    # dataframe.to_csv(file_path, index=True, header=True, chunksize=100000)
    dataframe.to_hdf(file_path, key="df", index=True)


def parse_frame_dates(frame):
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def loadDatasetsDirectly(
    dataset_path: str, dataset_info: DatasetInfo
) -> _LoadedDatasetPartNew:
    """
    Load the dataset directly from the files.
    """
    datasets = {}

    # TODO: Run checks as to confirm that the dataset_info.yaml information are right
    # eg. check start and end times

    data_dirs = ["demands", "flows", "levels", "pressures"]
    for data_dir in data_dirs:
        data_dir_in_dataset_dir = os.path.join(dataset_path, data_dir)
        if not os.path.exists(data_dir_in_dataset_dir):
            raise FileNotFoundError(
                f"Could not find the {data_dir} directory in the dataset directory ({dataset_path})"
            )
        datasets[data_dir] = {}

        csv_sensor_files = glob(os.path.join(data_dir_in_dataset_dir + "/" + "*.csv"))
        h5_sensor_files = glob(os.path.join(data_dir_in_dataset_dir + "/" + "*.h5"))

        if len(csv_sensor_files) > 0 and len(h5_sensor_files) > 0:
            raise Exception("Mix of csv and h5 files is not allowed")

        for sensor_readings_file in csv_sensor_files:
            logging.debug(f"Trying to load: {sensor_readings_file}")
            d = pd.read_csv(
                sensor_readings_file,
                index_col="Timestamp",
                chunksize=100000,
            )
            with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
                frames = list(executor.map(parse_frame_dates, d))

            sensor_readings = pd.concat(frames)

            datasets[data_dir][
                os.path.basename(sensor_readings_file)[:-4]
            ] = sensor_readings

        for sensor_readings_file in h5_sensor_files:
            logging.debug(f"Trying to load: {sensor_readings_file}")
            sensor_readings = pd.read_hdf(
                sensor_readings_file,
                key="df",
                index_col="Timestamp",
            )

            datasets[data_dir][
                os.path.basename(sensor_readings_file)[:-3]
            ] = sensor_readings

    date_columns = ["leak_time_start", "leak_time_end", "leak_time_peak"]
    datasets["leaks"] = pd.read_csv(
        os.path.join(dataset_path, "leaks.csv"),
        parse_dates=date_columns,
        # This fixes differences in dates, aswell as empty columns
        date_parser=lambda x: pd.to_datetime(x, utc=True),
    )

    return _LoadedDatasetPartNew(datasets)


class Dataset:
    """
    The dataset class.
    Loads data lazily and caches it.

        Config values Example:
            dataset:
                evaluation:
                    start: 2019-01-01 00:00
                    end: 2019-12-31 23:55
                training:
                    start: 2018-01-01 00:00
                    end: 2019-12-31 23:55
    See :class:`~ldimbenchmark.datasets.classes.DatasetInfo` for the all configuration option.

    This is an Adapter for loading the Datasets in the Dataset Format specified in the Design.
    It converts and represents the Low Level Interface for the LDIMBenchmark.
    """

    def __init__(self, path):
        """
        :param path: Path to the dataset folder
        """
        self.path = path
        self.__validation_file_name = ".validation"
        self.__validation_path = os.path.join(self.path, self.__validation_file_name)
        self.__dataset_info_file_name = "dataset_info.yaml"
        self.__dataset_info_file_path = os.path.join(
            self.path, self.__dataset_info_file_name
        )
        # Read dataset_info.yaml
        if os.path.isfile(self.__dataset_info_file_path):
            # file exists
            with open(self.__dataset_info_file_path) as f:
                self.info: DatasetInfo = yaml.safe_load(f)
                training_start = pd.to_datetime(
                    self.info["dataset"]["training"]["start"], utc=True
                )
                training_end = pd.to_datetime(
                    self.info["dataset"]["training"]["end"], utc=True
                )
                evaluation_start = pd.to_datetime(
                    self.info["dataset"]["evaluation"]["start"], utc=True
                )
                evaluation_end = pd.to_datetime(
                    self.info["dataset"]["evaluation"]["end"], utc=True
                )
                if training_start >= training_end:
                    raise Exception(
                        "Training start time is after or the same as training end time! (dataset_info.yaml)"
                    )
                if evaluation_start >= evaluation_end:
                    raise Exception(
                        "Evaluation start time is after or the same as evaluation end time! (dataset_info.yaml)"
                    )
                if training_start > evaluation_start:
                    raise Exception(
                        "Training start time is after evaluation start time! (dataset_info.yaml)"
                    )
                if training_end > evaluation_end:
                    raise Exception(
                        "Training end time is after evaluation end time! (dataset_info.yaml)"
                    )
                if evaluation_start < training_end:
                    raise Exception(
                        "Evaluation start time is before training end time! (dataset_info.yaml)"
                    )
                if evaluation_end < training_end:
                    raise Exception(
                        "Evaluation end time is before training end time! (dataset_info.yaml)"
                    )
                if training_start == training_end:
                    raise Exception(
                        "Training start time is equal to training end time! (dataset_info.yaml)"
                    )

                self.info["dataset"]["training"]["start"] = training_start
                self.info["dataset"]["training"]["end"] = training_end
                self.info["dataset"]["evaluation"]["start"] = evaluation_start
                self.info["dataset"]["evaluation"]["end"] = evaluation_end

                if "checksum" not in self.info:
                    logging.warn(
                        "No checksum for data integrity found in dataset_info.yaml"
                    )
                    self.info["checksum"] = None
                self.checksum = self.info["checksum"]

        else:
            raise Exception(
                f"No dataset_info.yaml file found! (not at: '{self.__dataset_info_file_path}')"
            )

        self.name = self.info["name"]
        self._update_id()
        self.__pickle_path = os.path.join(self.path, f"dataset-{self.id}.pickle")

        self.model = read_inpfile(os.path.join(self.path, self.info["inp_file"]))
        dma_path = os.path.join(self.path, "dmas.json")
        if os.path.isfile(dma_path):
            # file exists
            with open(dma_path, "r") as f:
                self.dmas = json.load(f)
        else:
            logging.warning(
                f"No dmas.json file found. Skipping loading of DMAs. ({self.name})"
            )
            self.dmas = None

    def _update_id(self, force=False):
        """
        Sets the id (hash) according to the information in "dataset_info.yaml"
        """

        derivations_hash = (
            "-"
            + hashlib.md5(
                json.dumps(self.info, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        )
        self.id = self.info["name"] + derivations_hash

    def _get_data_checksum(self, folder: str):
        """
        Generates a checksum for the data in the folder. (excluding the validation file and the dataset_info.yaml file)
        This should protect against "silent" changes in the dataset.
        """
        return dirhash(
            folder,
            "md5",
            ignore_hidden=True,
            excluded_files=[
                self.__dataset_info_file_name,
            ],
        )

    def _generate_checksum(self, folder: str):
        """
        Generates a checksum for the dataset folder. (excluding the validation file and the dataset_info.yaml file)
        This should protect against "silent" changes in the dataset.
        """

    def _generate_dataset_hash(self, folder: str):
        """
        Creates Hash for the whole dataset folder. (excluding only the validation file)
        This should protect against "silent" changes in the dataset and info file, while just looking something up.
        """
        new_hash = dirhash(folder, "md5", ignore_hidden=True)
        with open(os.path.join(folder, self.__validation_file_name), "w") as f:
            f.write(str(pd.to_datetime(datetime.now(), utc=True)) + "\n" + new_hash)

    def is_valid(self, check_consistency=False) -> bool:
        # Check the has of the whole dataset from time to time by writing the last timestamp
        # to a file and retrieving it before running the expensive hash function
        previous_info_hash = None
        try:
            with open(self.__validation_path, "r") as f:
                timestamp_file = f.readlines()
                last_timestamp = pd.to_datetime(timestamp_file[0], utc=True)
                previous_info_hash = timestamp_file[1]
        except (FileNotFoundError, IndexError):
            logging.warn("Dataset had no hashes for validation.")
            last_timestamp = None

        with open(self.__dataset_info_file_path, "r") as f:
            new_info_hash = hashlib.md5(f.read().encode("utf-8")).hexdigest()

        if previous_info_hash != None and previous_info_hash != new_info_hash:
            logging.warn(
                "dataset_info.yaml has changed since last validation! Check for changes and either revert them or make sure you want them by deleting the .validation file!"
            )
            return False
        previous_info_hash = new_info_hash

        if (
            last_timestamp is None
            or last_timestamp
            < pd.to_datetime(datetime.now(), utc=True) - pd.Timedelta("24 hour")
            or check_consistency
        ):
            logging.info("Checking data consistency...")

            data_checksum = self._get_data_checksum(self.path)
            if self.checksum != data_checksum:
                logging.info(
                    "Data checksum changed! Check the consistency of the dataset!"
                )
                return False

            with open(self.__validation_path, "w") as f:
                f.write(
                    str(pd.to_datetime(datetime.now(), utc=True))
                    + "\n"
                    + previous_info_hash
                )
        return True

    ######
    # Getters and Setters with direct lazy loading because they
    ######

    @property
    def pressures(self) -> DataFrame:
        if self.full_dataset_part is None or self.full_dataset_part.pressures is None:
            raise Exception("Call `loadData()` before accessing pressure data.")
        return self.full_dataset_part.pressures

    @pressures.setter
    def pressures(self, pressures: DataFrame):
        if self.full_dataset_part is None or self.full_dataset_part.pressures is None:
            raise Exception("Call `loadData()` before accessing pressure data.")
        self.full_dataset_part.pressures = pressures

    @property
    def demands(self) -> DataFrame:
        if self.full_dataset_part is None or self.full_dataset_part.demands is None:
            raise Exception("Call `loadData()` before accessing demand data.")
        return self.full_dataset_part.demands

    @demands.setter
    def demands(self, demands: DataFrame):
        if self.full_dataset_part is None or self.full_dataset_part.demands is None:
            raise Exception("Call `loadData()` before accessing demand data.")
        self.full_dataset_part.demands = demands

    @property
    def flows(self) -> DataFrame:
        if self.full_dataset_part is None or self.full_dataset_part.flows is None:
            raise Exception("Call `loadData()` before accessing flow data.")
        return self.full_dataset_part.flows

    @flows.setter
    def flows(self, flows: DataFrame):
        if self.full_dataset_part is None or self.full_dataset_part.flows is None:
            raise Exception("Call `loadData()` before accessing flow data.")
        self.full_dataset_part.flows = flows

    @property
    def levels(self) -> DataFrame:
        if self.full_dataset_part is None or self.full_dataset_part.levels is None:
            raise Exception("Call `loadData()` before accessing level data.")
        return self.full_dataset_part.levels

    @levels.setter
    def levels(self, levels: DataFrame):
        if self.full_dataset_part is None or self.full_dataset_part.levels is None:
            raise Exception("Call `loadData()` before accessing level data.")
        self.full_dataset_part.levels = levels

    @property
    def leaks(self) -> DataFrame:
        if self.full_dataset_part is None or self.full_dataset_part.leaks is None:
            raise Exception("Call `loadData()` before accessing leak data.")
        return self.full_dataset_part.leaks

    @leaks.setter
    def leaks(self, leaks: DataFrame):
        if self.full_dataset_part is None or self.full_dataset_part.leaks is None:
            raise Exception("Call `loadData()` before accessing leak data.")
        self.full_dataset_part.leaks = leaks

    def ensure_cached(self):
        if not os.path.isfile(self.__pickle_path):
            logging.info(f"Ensuring {self.id} is cached")
            self.loadData()

    def loadData(self):
        logging.debug(f"Loading dataset {self.id}")
        if not hasattr(self, "full_dataset_part"):
            if os.path.isfile(self.__pickle_path):
                try:
                    with open(self.__pickle_path, "rb") as f:
                        self.full_dataset_part = pickle.load(f)
                except Exception as e:
                    logging.error(f"Could not load pickle {self.id}! Regenerating...")

            if not hasattr(self, "full_dataset_part"):
                self.full_dataset_part = loadDatasetsDirectly(self.path, self.info)
                with open(self.__pickle_path, "wb") as f:
                    pickle.dump(self.full_dataset_part, f)
        logging.debug(f"Stopped loading dataset {self.id}")
        return self

    def loadBenchmarkData(self):
        """
        Preloads the dataset that contains the benchmark data for the training and evaluation dataset
        """
        logging.info("Start Loading benchmark data...")
        # Load Data
        if not hasattr(self, "train"):
            self.train = extractSubDataset(
                "training", self.info, self.full_dataset_part
            )
        if not hasattr(self, "evaluation"):
            self.evaluation = extractSubDataset(
                "evaluation", self.info, self.full_dataset_part
            )
        logging.info("Stop Loading benchmark data...")
        return self

    def getTrainingBenchmarkData(self):
        return BenchmarkData(
            pressures=self.train.pressures,
            demands=self.train.demands,
            flows=self.train.flows,
            levels=self.train.levels,
            model=self.model,
            dmas=self.dmas,
        )

    def getEvaluationBenchmarkData(self):
        return BenchmarkData(
            pressures=self.evaluation.pressures,
            demands=self.evaluation.demands,
            flows=self.evaluation.flows,
            levels=self.evaluation.levels,
            model=self.model,
            dmas=self.dmas,
        )

    def exportTo(self, folder: str):
        """
        Exports the dataset to a given folder
        """

        write_inpfile(self.model, os.path.join(folder, self.info["inp_file"]))

        # TODO: Probably better parallelize
        for sensor_type in ["pressures", "demands", "flows", "levels"]:
            os.makedirs(os.path.join(folder, sensor_type), exist_ok=True)
            filepaths = [
                os.path.join(folder, sensor_type, f"{sensor}.h5")
                for sensor in getattr(self, sensor_type).keys()
            ]
            values = getattr(self, sensor_type).values()
            # logging.debug(filepaths)
            with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
                # submit all tasks and get future objects
                futures = [
                    executor.submit(write_to_file, sensor, path)
                    for sensor, path in zip(values, filepaths)
                ]
                # process results from tasks in order of task completion
                for future in as_completed(futures):
                    future.result()

        self.leaks.to_csv(os.path.join(folder, "leaks.csv"))
        data_hash = self._get_data_checksum(folder)
        self.info["checksum"] = data_hash

        with open(os.path.join(folder, self.__dataset_info_file_name), "w") as f:
            yaml.dump(
                self.info, f, sort_keys=False, default_flow_style=None, Dumper=TSDumper
            )
        if os.path.exists(self.__pickle_path):
            os.remove(self.__pickle_path)
        self._update_id()
        self.is_valid()


def getTimeSliceOfDataset(
    dataset: Dict[str, DataFrame],
    start: datetime,
    end: datetime,
    logging_dataset_name: str,
    logging_sensor_type: str,
) -> DataFrame:
    """
    Get a time slice of a dataframe.
    """

    new_dataset_slice = {}
    for key in dataset:
        logging.debug(key)
        new_dataset_slice[key] = dataset[key].sort_index().loc[start:end]

    # If there is no data in the slice, but there is data in the dataset, then the start- and endtime are outside of the datapoint ranges.
    if len(new_dataset_slice) == 0 and len(dataset.keys()) != 0:
        logging.warning(
            f"No data from dataset '{logging_dataset_name}' with sensor type '{logging_sensor_type}' selected, because start- and endtime ({start} / {end}) are outside of the datapoint ranges."
        )
    return new_dataset_slice


def extractSubDataset(
    type: Literal["training", "evaluation"],
    config: DatasetInfo,
    full_data: _LoadedDatasetPartNew,
):
    if type != "training" and type != "evaluation":
        raise ValueError("type must be either 'training' or 'evaluation'")

    start_time = config["dataset"][type]["start"]
    end_time = config["dataset"][type]["end"]

    mask = (full_data.leaks["leak_time_start"] > start_time) & (
        full_data.leaks["leak_time_start"] < end_time
    )
    leaks = full_data.leaks[mask]

    return _LoadedDatasetPartNew(
        {
            "pressures": getTimeSliceOfDataset(
                full_data.pressures, start_time, end_time, config["name"], "pressures"
            ),
            "demands": getTimeSliceOfDataset(
                full_data.demands, start_time, end_time, config["name"], "demands"
            ),
            "flows": getTimeSliceOfDataset(
                full_data.flows, start_time, end_time, config["name"], "flows"
            ),
            "levels": getTimeSliceOfDataset(
                full_data.levels, start_time, end_time, config["name"], "levels"
            ),
            "leaks": leaks,
        }
    )


CACHE = {}


def __cache_dataset(id: str, dataset: Dataset):
    CACHE[id] = copy.deepcopy(dataset)


def __load_cached_dataset(id: str) -> Dataset:
    if id in CACHE:
        return copy.deepcopy(CACHE[id])


def __cached_dataset_exists(id: str) -> bool:
    if id in CACHE:
        return True
    return False
