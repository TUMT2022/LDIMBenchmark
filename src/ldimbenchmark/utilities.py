from typing import Dict, List
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


def resampleAndConcatSensors(
    sensors: Dict[str, DataFrame], resample_frequency="1T"
) -> DataFrame:
    """Resample all sensors to the same time interval and concatenate them into one single DataFrame"""

    concatenated_sensors = []
    for sensor_name, sensor_data in sensors.items():
        concatenated_sensors.append(sensor_data.resample(resample_frequency).mean())

    if len(concatenated_sensors) == 0:
        return pd.DataFrame()

    return pd.concat(
        concatenated_sensors,
        axis=1,
    ).interpolate(limit_direction="both")


def simplifyBenchmarkData(
    data: BenchmarkData, resample_frequency="1T"
) -> SimpleBenchmarkData:
    """Convert multiple timeseries to one timeseries"""

    return SimpleBenchmarkData(
        pressures=resampleAndConcatSensors(data.pressures, resample_frequency),
        demands=resampleAndConcatSensors(data.demands, resample_frequency),
        flows=resampleAndConcatSensors(data.flows, resample_frequency),
        levels=resampleAndConcatSensors(data.levels, resample_frequency),
        model=data.model,
        dmas=data.dmas,
    )
