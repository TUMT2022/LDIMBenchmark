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


def resampleAndConcatSensors(
    sensors: Dict[str, DataFrame], should_have_value_count: int, resample_frequency="1T"
) -> DataFrame:
    """
    Resample all sensors to the same time interval and concatenate them into one single DataFrame

    value_count: The number of values that should be in the resulting DataFrame, if the number of values in sensors is less than value_count,
    the DataFrame will be padded with NaNs
    """

    concatenated_sensors = []
    for sensor_name, sensor_data in sensors.items():
        new_data = sensor_data.resample(resample_frequency).mean()
        if len(new_data) > len(sensor_data):
            logging.warning(
                "Upsampling data, this might result in one off errors later on. Consider settings 'resample_frequency' to a bigger timeframe."
            )

        # Making sure the DataFrame has the amount of values it should have
        missing_values_count = should_have_value_count - len(new_data)
        if missing_values_count > 0:
            missing_timedelta = pd.Timedelta(resample_frequency) * missing_values_count

            missing_dates = pd.DataFrame(
                index=pd.date_range(
                    new_data.iloc[-1].name,
                    new_data.iloc[-1].name + missing_timedelta,
                    freq=resample_frequency,
                )
            )
            missing_dates[0] = np.NaN
            missing_dates.columns = new_data.columns
            missing_dates
            new_data = new_data.combine_first(missing_dates)

        concatenated_sensors.append(new_data)

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

    max_values = 0
    for datasets in [data.pressures, data.demands, data.flows, data.levels]:
        for key in datasets.keys():
            max_values = max(max_values, len(datasets[key]))

    return SimpleBenchmarkData(
        pressures=resampleAndConcatSensors(
            data.pressures, max_values, resample_frequency
        ),
        demands=resampleAndConcatSensors(data.demands, max_values, resample_frequency),
        flows=resampleAndConcatSensors(data.flows, max_values, resample_frequency),
        levels=resampleAndConcatSensors(data.levels, max_values, resample_frequency),
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
