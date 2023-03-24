import logging
from typing import Dict, List
import numpy as np
from pandas import DataFrame
from ldimbenchmark.classes import BenchmarkData
from wntr.network import WaterNetworkModel
import pandas as pd


class SimpleBenchmarkData:
    """
    Representation of simplified Benchmark Dataset

    The main differences are:
    - sensors are not separated into single Time series, but are already aligned and in one single DataFrame

    This has the drawback that they are already resampled to the same time interval.
    """

    def __init__(
        self,
        pressures: DataFrame,
        demands: DataFrame,
        flows: DataFrame,
        levels: DataFrame,
        model: WaterNetworkModel,
        dmas: List[str],
    ):
        """
        Initialize the BenchmarkData object.
        """
        self.pressures = pressures
        """Pressures of the System."""
        self.demands = demands
        """Demands of the System."""
        self.flows = flows
        """Flows of the System."""
        self.levels = levels
        """Levels of the System."""
        self.model = model
        """Model of the System (INP)."""
        self.dmas = dmas
        """
        District Metered Areas
        Dictionary with names of the areas as key and list of WN nodes as value.
        """
        self.metadata = {}
        """Metadata of the System. e.g. Metering zones and included sensors."""


def resampleSensors(
    sensors: Dict[str, DataFrame], resample_frequency="1T"
) -> DataFrame:
    """
    Resample all sensors to the same time interval and concatenate them into one single DataFrame
    """

    for sensor_name, sensor_data in sensors.items():
        new_data = sensor_data.resample(resample_frequency).mean()
        if len(new_data) > len(sensor_data):
            logging.warning(
                f"Upsampling data of sensor '{sensor_name}', this might result in one off errors later on. Consider settings 'resample_frequency' to a bigger timeframe. ({len(sensor_data)} to {len(new_data)} datapoints)"
            )
        sensors[sensor_name] = new_data
    return sensors


def concatAndInterpolateSensors(
    sensors: Dict[str, DataFrame],
    should_have_value_count: int = None,
    resample_frequency="1T",
) -> DataFrame:
    """
    Concatenate all sensors to a single DataFrame and interpolate the missing values

    should_have_value_count: if the number of values in sensors is less than value_count,
    the DataFrame will be padded with NaNs
    """

    concatenated_sensors = []
    for sensor_name, sensor_data in sensors.items():
        if should_have_value_count is not None:
            # Making sure the DataFrame has the amount of values it should have
            logging.warning(f"SENSOR: {sensor_name}")
            logging.warning(f"MAX VALUES: {should_have_value_count}")
            missing_values_count = should_have_value_count - len(sensor_data)
            logging.warning(f"MISSING VALUES: {missing_values_count}")
            if missing_values_count > 0:
                missing_timedelta = (
                    pd.Timedelta(resample_frequency) * missing_values_count
                )

                missing_dates = pd.DataFrame(
                    index=pd.date_range(
                        sensor_data.iloc[-1].name,
                        sensor_data.iloc[-1].name + missing_timedelta,
                        freq=resample_frequency,
                    )
                )
                missing_dates[0] = np.NaN
                missing_dates.columns = sensor_data.columns
                missing_dates
                sensor_data = sensor_data.combine_first(missing_dates)

        concatenated_sensors.append(sensor_data)

    if len(concatenated_sensors) == 0:
        return pd.DataFrame()

    return pd.concat(
        concatenated_sensors,
        axis=1,
    ).interpolate(limit_direction="both")


def simplifyBenchmarkData(
    data: BenchmarkData, resample_frequency="1T", force_same_length=False
) -> SimpleBenchmarkData:
    """Convert multiple timeseries to one timeseries"""

    resampled_pressures = resampleSensors(data.pressures, resample_frequency)
    resampled_demands = resampleSensors(data.demands, resample_frequency)
    resampled_flows = resampleSensors(data.flows, resample_frequency)
    resampled_levels = resampleSensors(data.levels, resample_frequency)

    if force_same_length:
        max_values = 0
        for datasets in [
            resampled_pressures,
            resampled_demands,
            resampled_flows,
            resampled_levels,
        ]:
            for key in datasets.keys():
                max_values = max(max_values, len(datasets[key]))
    else:
        max_values = None

    return SimpleBenchmarkData(
        pressures=concatAndInterpolateSensors(
            resampled_pressures, max_values, resample_frequency
        ),
        demands=concatAndInterpolateSensors(
            resampled_demands, max_values, resample_frequency
        ),
        flows=concatAndInterpolateSensors(
            resampled_flows, max_values, resample_frequency
        ),
        levels=concatAndInterpolateSensors(
            resampled_levels, max_values, resample_frequency
        ),
        model=data.model,
        dmas=data.dmas,
    )


def getDmaSpecificData(data: SimpleBenchmarkData, sensors: List[str]):
    return SimpleBenchmarkData(
        pressures=data.pressures.loc[:, data.pressures.columns.isin(sensors)],
        demands=data.demands.loc[:, data.demands.columns.isin(sensors)],
        flows=data.flows.loc[:, data.flows.columns.isin(sensors)],
        levels=data.levels.loc[:, data.levels.columns.isin(sensors)],
        # TODO: This should probably be better handled:
        model=data.model,
        dmas=data.dmas,
    )


import os
import hashlib
import re
from joblib import Parallel, delayed

HASH_FUNCS = {
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512,
}


def dirhash(
    dirname,
    hashfunc="md5",
    excluded_files=None,
    ignore_hidden=False,
    followlinks=False,
    parallel=False,
):
    """
    Function for deterministically creating a single hash for a directory of files,
    taking into account only file contents and not filenames.
    From https://raw.githubusercontent.com/to-mc/checksumdir/0ec7096945e4778c23e16fbfe5183fe8dc62a21c/checksumdir/__init__.py

    """
    hash_func = HASH_FUNCS.get(hashfunc)
    if not hash_func:
        raise NotImplementedError("{} not implemented.".format(hashfunc))

    if not excluded_files:
        excluded_files = []

    if not os.path.isdir(dirname):
        raise TypeError("{} is not a directory.".format(dirname))
    fileslist = []
    for root, dirs, files in os.walk(dirname, topdown=True, followlinks=followlinks):
        if ignore_hidden:
            if not re.search(r"/\.", root):
                fileslist.extend(
                    [
                        os.path.join(root, f)
                        for f in files
                        if not f.startswith(".")
                        and not re.search(r"/\.", f)
                        and f not in excluded_files
                    ]
                )
        else:
            fileslist.extend(
                [os.path.join(root, f) for f in files if f not in excluded_files]
            )

    if parallel:
        hashvalues = Parallel(n_jobs=100, prefer="threads")(
            delayed(_filehash)(f, hash_func) for f in fileslist
        )
    else:
        hashvalues = [_filehash(f, hash_func) for f in fileslist]

    return _reduce_hash(hashvalues, hash_func)


def _filehash(filepath, hashfunc):
    hasher = hashfunc()
    blocksize = 64 * 1024
    try:
        with open(filepath, "rb") as fp:
            while True:
                data = fp.read(blocksize)
                if not data:
                    break
                hasher.update(data)
    except:
        pass
        # "The file %s no longer exists."%filepath

    return hasher.hexdigest()


def _reduce_hash(hashlist, hashfunc):
    hasher = hashfunc()
    for hashvalue in sorted(hashlist):
        hasher.update(hashvalue.encode("utf-8"))
    return hasher.hexdigest()
