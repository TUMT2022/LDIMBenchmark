from ldimbenchmark import (
    LDIMMethodBase,
    BenchmarkLeakageResult,
    MethodMetadata,
    Hyperparameter,
)
from ldimbenchmark.classes import BenchmarkData, MethodMetadataDataNeeded

from datetime import timedelta


import numpy as np
import pandas as pd

from ldimbenchmark.utilities import simplifyBenchmarkData
import math


class MNF(LDIMMethodBase):
    """
    MNF - Minimum Night Flow

    Method from KIOS Research Team

    Link: https://github.com/KIOS-Research/LeakDB/tree/master/CCWI-WDSA2018/Detection%20Algorithms/MNF

    Changelog:
    1.0.0 - Version from KIOS Research Team
    1.1.0 - Add option for resample_frequency
    1.2.0 - Run MNF for each sensor
    1.3.0 - Add option for sensor_treatment
    1.4.0 - Add option to set "night flow" interval and start time
    """

    def __init__(self):
        super().__init__(
            name="mnf",
            version="1.4.0",
            metadata=MethodMetadata(
                data_needed=MethodMetadataDataNeeded(
                    pressures="ignored",
                    flows="necessary",
                    levels="ignored",
                    model="ignored",
                    demands="ignored",
                    structure="ignored",
                ),
                capability="detect",
                paradigm="offline",
                extra_benefits="none",
                hyperparameters=[
                    Hyperparameter(
                        name="resample_frequency",
                        description="Time-frequency for resampling the data. e.g. '1T' for one minute, '1H' for one hour, '1D' for one day.",
                        value_type=str,
                        default="5T",
                    ),
                    Hyperparameter(
                        name="window",
                        description="Window size for the sliding window in units of 'night_flow_interval'",
                        value_type=int,
                        default=10,
                        min=1,
                        max=365,
                    ),
                    Hyperparameter(
                        name="gamma",
                        description="Threshold to raise a leak event",
                        value_type=float,
                        default=0.1,
                        min=0.0,
                        max=1.0,
                    ),
                    Hyperparameter(
                        name="sensor_treatment",
                        description="How to treat multiple flow sensors. 'each' for applying the method on each sensor individually, 'first' for using only the first sensor, 'sum' for applying the method on the sum of all sensors",
                        value_type=str,
                        default="each",
                        options=["each", "first", "sum"],
                    ),
                    Hyperparameter(
                        name="night_flow_interval",
                        description="Interval for the night flow span, normally 1440T for one day, but could also be 60T for one hour",
                        value_type=str,
                        default="1440T",
                    ),
                    Hyperparameter(
                        name="night_flow_start",
                        description="Start time for the night flow interval. Normally mid of the day, but could also be '2023-07-20 20:53:46.954726'. Only the Time section is considered.",
                        value_type=str,
                        default="2023-01-01 12:00:00",
                    ),
                ],
            ),
        )

    def prepare(self, train_data: BenchmarkData = None):
        # self.train_Data = train_data
        if train_data != None:
            self.simple_train_data = simplifyBenchmarkData(
                train_data,
                resample_frequency=self.hyperparameters["resample_frequency"],
            )
        else:
            self.simple_train_data = None

    def detect_offline(self, evaluation_data: BenchmarkData):
        night_flow_interval = pd.Timedelta(self.hyperparameters["night_flow_interval"])
        window_steps = self.hyperparameters["window"]
        window = night_flow_interval * window_steps
        gamma: float = self.hyperparameters["gamma"]

        simple_evaluation_data = simplifyBenchmarkData(
            evaluation_data,
            resample_frequency=self.hyperparameters["resample_frequency"],
        )

        # If the data is too short, return an empty list
        if (
            simple_evaluation_data.flows.index[-1]
            - simple_evaluation_data.flows.index[0]
            < 3 * window
        ):
            return []

        interval_start = pd.to_datetime(
            np.datetime64(self.hyperparameters["night_flow_start"])
        )
        start_time = simple_evaluation_data.flows[
            (simple_evaluation_data.flows.index.hour == interval_start.hour)
            & (simple_evaluation_data.flows.index.minute == interval_start.minute)
            & (simple_evaluation_data.flows.index.second == interval_start.second)
        ].index[0]

        end_time = simple_evaluation_data.flows[
            (simple_evaluation_data.flows.index.hour == interval_start.hour)
            & (simple_evaluation_data.flows.index.minute == interval_start.minute)
            & (simple_evaluation_data.flows.index.second == interval_start.second)
        ].index[-1]

        end_time = start_time + (
            night_flow_interval
            * math.floor(
                (simple_evaluation_data.flows.index[-1] - start_time)
                / night_flow_interval
            )
        )

        all_flows = simple_evaluation_data.flows.loc[
            (simple_evaluation_data.flows.index >= start_time)
            & (simple_evaluation_data.flows.index < end_time)
        ]

        if self.simple_train_data:
            # Use training data to set up the window, so we can start with the evaluation data
            previous_data = self.simple_train_data.flows
            previous_start_time = previous_data[
                (previous_data.index.hour == interval_start.hour)
                & (previous_data.index.minute == interval_start.minute)
                & (previous_data.index.second == interval_start.second)
            ].index[0]

            previous_end_time = previous_start_time + (
                night_flow_interval
                * math.floor(
                    (previous_data.index[-1] - previous_start_time)
                    / night_flow_interval
                )
            )

            mask = (previous_data.index >= previous_start_time) & (
                previous_data.index < previous_end_time
            )
            previous_data = previous_data.loc[mask]

            all_flows = pd.concat([previous_data, all_flows], axis=0)

        # TODO: For now lets say it starts at noon
        night_flow_interval_end = start_time + night_flow_interval

        # better: all_flows.groupby(all_flows.index.date).size()
        entries_per_interval = (
            (all_flows.index > start_time)
            & (all_flows.index <= night_flow_interval_end)
        ).sum()

        days = int(all_flows.shape[0] / entries_per_interval)

        results = []
        # all_flows = pd.DataFrame(all_flows.sum(axis=1))
        if self.hyperparameters["sensor_treatment"] == "first":
            all_flows = all_flows[all_flows.columns[0:1]]
        elif self.hyperparameters["sensor_treatment"] == "sum":
            all_flows = pd.DataFrame(all_flows.sum(axis=1))

        for sensor in all_flows.columns:
            flows_array = all_flows[sensor].to_numpy()

            # Error here
            # array of size 58175 into shape (201, 288)
            reshaped = np.reshape(flows_array, (days, entries_per_interval))

            min_flows = reshaped.min(axis=1)

            labels = np.zeros(days)
            # start the search for leaks at time window + first day
            current_analysis_frame = window_steps + 1
            while current_analysis_frame < days:
                min_window = min(
                    min_flows[
                        current_analysis_frame - window_steps : current_analysis_frame
                    ]
                )
                residual = min_flows[current_analysis_frame] - min_window

                # If residual is greater than gamma times the minimum window flow
                if residual > min_window * gamma:
                    labels[current_analysis_frame] = 1

                current_analysis_frame += 1

            full_labels = np.repeat(labels, entries_per_interval)

            # Pattern search for change in labels
            searchval = [0, 1]
            leaks = all_flows.index[
                np.concatenate(
                    (
                        (full_labels[:-1] == searchval[0])
                        & (full_labels[1:] == searchval[1]),
                        [False],
                    )
                )
            ]

            for leak_start in leaks:
                results.append(
                    BenchmarkLeakageResult(
                        leak_pipe_id=sensor,
                        leak_time_start=leak_start,
                        leak_time_end=leak_start,
                        leak_time_peak=leak_start,
                        leak_area=0.0,
                        leak_diameter=0.0,
                        leak_max_flow=0.0,
                    )
                )
        return results

        # for i=0; i<days; i++:
        #     min_window = min(min_flows[i:i+window.days])
        #     if min_flows[current_analysis_day] - min_window > min_window * gamma:
        # start_date.
        # % LScFlows: vector of all measurements
        # % 365 * 24 * 2 (2 measurements per hour)
        # %LScFlows = zeros(17520, 1);
        # %LScFlows = randn(17520,1);
        # %gamma = 0.1;
        # %t1 = datetime(2013,1,1,8,0,0);
        # %t2 = t1 + days(365) - minutes(30);
        # %timeStamps = t1:minutes(30):t2;

        #     %% MNF code
        #     w=10; % window size
        #     k = 1:w; % window: day indices
        #     Labels_Sc=[];

        #     reshaped = reshape(LScFlows,48,365);
        #     % Shape into day sized vectors

        #     MNF = min(reshape(LScFlows,48,365));
        #     %get minimum flows per day

        #     % start the search for leaks at time window + first day
        #     for j=(w+1):365
        #         % get MNF of the 10 day window
        #         minMNFW = min(MNF(k));
        #         % get residual of current day MNF and minmum MNF of the window
        #         e = MNF(j)-minMNFW;

        #         % compare residual against minmum night flow threshold
        #         if e>minMNFW*gamma
        #             % set label of current day
        #             Labels_Sc(j) = 1;
        #         else
        #             % set label of current day
        #             Labels_Sc(j) = 0;
        #             % move window one day forward, e.g. [1:10] to [2:11]
        #             k(w+1) = j;
        #             k(1) = [];
        #         end
        #     end

        #     Labels_Sc_Final1 = [];
        #     j=48; % j=number of measurements per day
        #     % for each day
        #     for d=1:size(Labels_Sc,2)
        #         % Scale Labels to measurements vector by applying the daily label
        #         % to each measurement
        #         Labels_Sc_Final1(j-47:j,1)=Labels_Sc(d);
        #         j = j+48;
        #     end

        #     clear Labels_Sc
        #     % Combine labels and timestamps?
        #     Labels_Sc = [datestr(timeStamps, 'yyyy-mm-dd HH:MM') repmat(', ',length(timeStamps),1) num2str(repmat(Labels_Sc_Final1, 1))];
        #     Labels_Sc = cellstr(Labels_Sc);

    def detect_online(self, evaluation_data):
        pass
