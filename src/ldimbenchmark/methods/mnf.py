from ldimbenchmark import (
    LDIMMethodBase,
    BenchmarkLeakageResult,
    MethodMetadata,
    Hyperparameter,
)
from ldimbenchmark.classes import BenchmarkData

from datetime import timedelta
from sklearn.linear_model import LinearRegression
import sklearn
import pickle
import math
from pandas import Timestamp


import numpy as np
import pandas as pd

from ldimbenchmark.utilities import simplifyBenchmarkData


class MNF(LDIMMethodBase):
    """
    Minimum Night Flow Method from
    https://github.com/KIOS-Research/LeakDB/tree/master/CCWI-WDSA2018/Detection%20Algorithms/MNF
    """

    def __init__(self):
        super().__init__(
            name="mnf",
            version="1.2",
            metadata=MethodMetadata(
                data_needed=["flow"],
                hyperparameters=[
                    Hyperparameter(
                        name="resample_frequency",
                        description="Time frequency for resampling the data. e.g. '1T' for 1 minute, '1H' for 1 hour, '1D' for 1 day.",
                        value_type=str,
                        default="5T",
                    ),
                    Hyperparameter(
                        name="window",
                        description="Window size for the sliding window in days",
                        value_type=int,
                        default=10,
                        min=1,
                        max=365,
                    ),
                    Hyperparameter(
                        name="gamma",
                        description="Threshold to raise an alert",
                        value_type=float,
                        default=0.1,
                        min=0.0,
                        max=1.0,
                    ),
                ],
                # TODO: more attributes?
                mimum_dataset_size=365,  # in days to match datasets?
                can_resample=False,  # Whether the method can resample the data itself or if it should be resampled before?
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
        window = pd.Timedelta(days=self.hyperparameters["window"])
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

        day_counts = simple_evaluation_data.flows.groupby(
            simple_evaluation_data.flows.index.date
        ).size()

        if day_counts[day_counts.index[0]] != day_counts[day_counts.index[1]]:
            first_full_day = day_counts.index[1]
        else:
            first_full_day = day_counts.index[0]

        start_date = pd.to_datetime(first_full_day, utc=True).replace(
            hour=12, minute=0, second=0, microsecond=0, nanosecond=0
        )

        if day_counts[day_counts.index[-1]] != day_counts[day_counts.index[-2]]:
            last_full_day = day_counts.index[-2]
        else:
            last_full_day = day_counts.index[-1]

        end_date = pd.to_datetime(last_full_day, utc=True).replace(
            hour=12, minute=0, second=0, microsecond=0, nanosecond=0
        )

        all_flows = simple_evaluation_data.flows.loc[
            (simple_evaluation_data.flows.index >= start_date)
            & (simple_evaluation_data.flows.index < end_date)
        ]

        if self.simple_train_data:
            # Use training data to set up the window, so we can start with the evaluation data
            previous_data = self.simple_train_data.flows
            previous_data_day_counts = previous_data.groupby(
                previous_data.index.date
            ).size()

            if (
                previous_data_day_counts[previous_data_day_counts.index[-1]]
                != previous_data_day_counts[previous_data_day_counts.index[-2]]
            ):
                last_full_day = previous_data_day_counts.index[-2]
            else:
                last_full_day = previous_data_day_counts.index[-1]

            previous_end_date = pd.to_datetime(last_full_day, utc=True).replace(
                hour=12, minute=0, second=0, microsecond=0, nanosecond=0
            )
            if previous_data.index[-1] - previous_data.index[0] < 3 * window:
                if (
                    previous_data_day_counts[previous_data_day_counts.index[0]]
                    != previous_data_day_counts[previous_data_day_counts.index[1]]
                ):
                    first_full_day = previous_data_day_counts.index[1]
                else:
                    first_full_day = previous_data_day_counts.index[0]

                previous_start_date = pd.to_datetime(first_full_day, utc=True).replace(
                    hour=12, minute=0, second=0, microsecond=0, nanosecond=0
                )
            else:
                previous_start_date = previous_end_date - window

            mask = (previous_data.index >= previous_start_date) & (
                previous_data.index < previous_end_date
            )
            previous_data = self.simple_train_data.flows.loc[mask]

            all_flows = pd.concat([previous_data, all_flows], axis=0)

        # TODO: For now lets say it starts at noon
        hour_24_end = start_date + timedelta(days=1)

        # better:
        entries_per_day = (
            (all_flows.index > start_date) & (all_flows.index <= hour_24_end)
        ).sum()

        days = int(all_flows.shape[0] / entries_per_day)

        results = []
        # all_flows = pd.DataFrame(all_flows.sum(axis=1))
        for sensor in all_flows.columns:
            flows_array = all_flows[sensor].to_numpy()

            # Error here
            # array of size 58175 into shape (201, 288)
            reshaped = np.reshape(flows_array, (days, entries_per_day))

            min_flows = reshaped.min(axis=1)

            labels = np.zeros(days)
            # start the search for leaks at time window + first day
            current_analysis_day = window.days + 1
            while current_analysis_day < days:
                min_window = min(
                    min_flows[current_analysis_day - window.days : current_analysis_day]
                )
                residual = min_flows[current_analysis_day] - min_window

                # If residual is greater than gamma times the minimum window flow
                if residual > min_window * gamma:
                    labels[current_analysis_day] = 1

                current_analysis_day += 1

            full_labels = np.repeat(labels, entries_per_day)

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
