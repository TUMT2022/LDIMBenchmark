import logging
from ldimbenchmark.classes import LDIMMethodBase, MethodMetadata, Hyperparameter
from ldimbenchmark import (
    BenchmarkData,
    BenchmarkLeakageResult,
)
from ldimbenchmark.methods.utils.cusum import cusum

from datetime import timedelta
from sklearn.linear_model import LinearRegression
import sklearn
import pickle
import math
from typing import List
import numpy as np
import pandas as pd
import os

from ldimbenchmark.utilities import (
    SimpleBenchmarkData,
    getDmaSpecificData,
    simplifyBenchmarkData,
)


class Ref_node:
    def __init__(self, name):
        self.name = name

    def set_models(self, models):
        self._models_Reg = models


class SCADA_data:
    def __init__(self, pressures=None, flows=None, demands=None, levels=None):
        self.pressures = pressures
        self.flows = flows
        self.demands = demands
        self.levels = levels


class LILA(LDIMMethodBase):
    """
    LILA - Leakage Identification and Localization Algorithm

    Link: https://github.com/SWN-group-at-TU-Berlin/LILA/tree/main

    Version History:
      0.1.0: Initial version from the authors, with performance tweaks
      0.2.0: Refactored and added DMA specific analysis
      0.2.1: Make sure resampled sensor data can be used by LILA
    """

    def __init__(self):
        super().__init__(
            name="lila",
            version="0.2.0",
            metadata=MethodMetadata(
                data_needed=["pressures", "flows"],
                hyperparameters=[
                    Hyperparameter(
                        name="leakfree_time_start",
                        description="Start date of a leakfree period of time inside the dataset. Format should be 'YYYY-MM-DD'. Leaving it empty will use the whole dataset.",
                        value_type=str,
                    ),
                    Hyperparameter(
                        name="leakfree_time_stop",
                        description="End date of a leakfree period of time inside the dataset. Format should be 'YYYY-MM-DD'. Leaving it empty will use the whole dataset.",
                        value_type=str,
                    ),
                    Hyperparameter(
                        name="resample_frequency",
                        description="Time frequency for resampling the data. e.g. '1T' for 1 minute, '1H' for 1 hour, '1D' for 1 day.",
                        value_type=str,
                        default="5T",
                    ),
                    Hyperparameter(
                        name="est_length",
                        description="Length of the estimation period in hours",
                        default=72,  # 3 days
                        value_type=int,
                        max=8760,  # 1 year
                        min=1,
                    ),
                    Hyperparameter(
                        name="est_length",
                        description="Length of the estimation period in hours",
                        default=72,  # 3 days
                        value_type=int,
                        max=8760,  # 1 year
                        min=1,
                    ),
                    Hyperparameter(
                        name="C_threshold",
                        description="Threshold for the CUSUM statistic",
                        default=3.0,
                        value_type=float,
                        max=10.0,
                        min=0.0,
                    ),
                    Hyperparameter(
                        name="delta",
                        description="Delta for the CUSUM statistic",
                        default=4.0,
                        value_type=float,
                        max=10.0,
                        min=0.0,
                    ),
                    Hyperparameter(
                        name="dma_specific",
                        description="Whether the method should be executed on each dma or on the whole dataset at once",
                        default=False,
                        value_type=bool,
                    ),
                    Hyperparameter(
                        name="default_flow_sensor",
                        description="Flow Sensor used for dma unspecific analysis or if no flow sensor is available for a dma.",
                        required=True,
                        value_type=str,
                    ),
                ],
            ),
        )
        self.trained = False

    # TODO: Add DMA specific implementation (and hyperparameters)
    # {
    #                 "P-01": "PUMP_1",
    #                 "J-02": "PUMP_1",
    #                 "Inflow [l/s]": "PUMP_1",
    #                 "31664": "PUMP_1",
    #                 "wNode_1": "PUMP_1",
    #             }
    def _train(
        self,
        simple_train_data: SimpleBenchmarkData,
        start_time,
        end_time,
        dma: str,
    ):
        self.trained = True

        nodes = simple_train_data.pressures.keys()

        N = len(nodes)

        if not hasattr(self, "K0"):
            self.K0 = {}
            self.K1 = {}
            self.Kd = {}
        self.K0[dma] = np.zeros((N, N))
        self.K1[dma] = np.zeros((N, N))
        self.Kd[dma] = np.zeros((N, N))

        for i, node in enumerate(nodes):
            X_tr = np.concatenate(
                [
                    simple_train_data.pressures[node]
                    .loc[start_time:end_time]
                    .to_numpy()
                    .reshape(-1, 1),
                    simple_train_data.flows["PUMP_1"]
                    .loc[start_time:end_time]
                    .to_numpy()
                    .reshape(-1, 1),
                ],
                axis=1,
            )

            for j, node_cor in enumerate(nodes):
                y_tr = (
                    simple_train_data.pressures[node_cor]
                    .loc[start_time:end_time]
                    .to_numpy()
                    .reshape(-1, 1)
                )
                model = LinearRegression()
                model.fit(X_tr, y_tr)

                self.K0[dma][i, j] = model.intercept_[0]
                self.K1[dma][i, j] = model.coef_[0][0]
                self.Kd[dma][i, j] = model.coef_[0][1]

    def _detect(self, data: SimpleBenchmarkData, dma_key: str):
        nodes = data.pressures.keys()

        # Leak Analysis Function
        # nodes - List of Nodes which should be.
        # scada - Dataset which holds flow and pressure data.
        # cor_time_frame - Time Frame for where there is no leak.
        # def leak_analysis(nodes, scada_data, cor_time_frame):
        N = len(nodes)
        # T = Timestamps
        T = data.pressures.shape[0]
        P = data.pressures[nodes].values
        V = data.flows["PUMP_1"].values

        np.fill_diagonal(self.K0[dma_key], 0)
        np.fill_diagonal(self.K1[dma_key], 1)
        np.fill_diagonal(self.Kd[dma_key], 0)

        # If there is an error look up the problem at:
        # https://github.com/numpy/numpy/issues/23244
        e = (
            np.multiply.outer(self.K0[dma_key], np.ones(T))
            + np.multiply.outer(self.Kd[dma_key], V)
            + np.multiply.outer(self.K1[dma_key], np.ones(T))
            * np.multiply.outer(P, np.ones(N)).transpose(1, 2, 0)
            - np.multiply.outer(P, np.ones(N)).transpose(2, 1, 0)
        )

        # # Why?
        # if 'n215' in nodes:
        #     e[nodes.index('n215'), :, :] *= 0.02

        # Faster:
        e_count = e.copy()
        e_count[e_count < 0] = 0
        e_count[e_count > 0] = 1
        # Find Sensor Indexes which are most affected
        # As the previous implementation used quicksort (with descending order) we have to use this beautiful hack
        max_affected_sensors_index = np.abs(
            (e_count.sum(axis=0)[::-1]).argsort(kind="quicksort", axis=0) - (N - 1)
        )[-1, :]
        # There are better ways, which need to be tested.
        # # 1. Use stable sort
        # max_affected_sensors_index = e_count.sum(
        #     axis=0).argsort(kind='stable', axis=0)[-1, :]

        # # 2. Simply use argmax
        # max_affected_sensors_index = e_count.sum(axis=0).argmax(axis=0)

        # Select Values of most affected sensors (n, T)
        max_affected_sensor_values = np.take_along_axis(
            e,
            np.expand_dims(np.expand_dims(max_affected_sensors_index, axis=0), axis=0),
            axis=1,
        ).squeeze()
        norm_values = np.linalg.norm(max_affected_sensor_values, axis=0)
        res = np.zeros((N, T))
        np.put_along_axis(
            res, np.expand_dims(max_affected_sensors_index, axis=0), norm_values, axis=0
        )

        MRE = pd.DataFrame(res.T, index=data.pressures.index, columns=nodes)
        leaks, cusum_data = cusum(
            MRE,
            C_thr=self.hyperparameters["C_threshold"],
            delta=self.hyperparameters["delta"],
            est_length=self.hyperparameters["est_length"],
        )

        if self.debug:
            MRE.to_csv(os.path.join(self.additional_output_path, "mre.csv"))
            # print(MRE)
            # for sensor in MRE.columns:
            #     MRE_single = MRE[[sensor]]
            #     CUSUM_DATA = cusum(MRE_single, est_length="1 minute")
            #     # print(sensor)
            #     print(CUSUM_DATA[0])
            # print(leaks)
            # print(rawdata)
            cusum_data.to_csv(os.path.join(self.additional_output_path, "cusum.csv"))

        # Overall MRE is not good for detection, so we just keep these Nodes as Sensors to Consider in the next Step
        return leaks

    def prepare(self, training_data: BenchmarkData = None) -> None:
        if training_data != None:
            simple_training_data = simplifyBenchmarkData(
                training_data,
                resample_frequency=self.hyperparameters["resample_frequency"],
                force_same_length=True,
            )

            start_time = pd.to_datetime(self.hyperparameters["leakfree_time_start"])
            end_time = pd.to_datetime(self.hyperparameters["leakfree_time_stop"])
            if self.hyperparameters["dma_specific"]:
                for dma in training_data.dmas.keys():
                    dma_specific_training_data = getDmaSpecificData(
                        simple_training_data, training_data.dmas[dma]
                    )
                    if len(dma_specific_training_data[dma].flows.columns) == 0:
                        dma_specific_training_data[dma].flows[
                            "PUMP_1"
                        ] = simple_training_data.flows[
                            self.hyperparameters["default_flow_sensor"]
                        ]
                    else:
                        dma_specific_training_data[dma].flows[
                            "PUMP_1"
                        ] = dma_specific_training_data[dma].flows.sum(axis=1)
                    # dma_specific_training_data.flows[
                    #     "PUMP_1"
                    # ] = dma_specific_training_data.flows.sum(axis=1)

                    self._train(
                        dma_specific_training_data, start_time, end_time, dma=dma
                    )

            else:
                if self.hyperparameters["default_flow_sensor"] == "sum":
                    simple_training_data.flows[
                        "PUMP_1"
                    ] = simple_training_data.flows.sum(axis=1)
                else:
                    if (
                        self.hyperparameters["default_flow_sensor"]
                        in simple_training_data.flows.columns
                    ):
                        simple_training_data.flows[
                            "PUMP_1"
                        ] = simple_training_data.flows[
                            self.hyperparameters["default_flow_sensor"]
                        ]
                    else:
                        raise Exception(
                            "No 'default_flow_sensor' hyperparameter given or the column does not exist."
                        )
                self._train(simple_training_data, start_time, end_time, dma="main")

    def detect_offline(
        self, evaluation_data: BenchmarkData
    ) -> List[BenchmarkLeakageResult]:
        simple_evaluation_data = simplifyBenchmarkData(
            evaluation_data,
            resample_frequency=self.hyperparameters["resample_frequency"],
            force_same_length=True,
        )

        dma_specific_data = {}
        if self.hyperparameters["dma_specific"]:
            for dma in evaluation_data.dmas.keys():
                dma_specific_data[dma] = getDmaSpecificData(
                    simple_evaluation_data, evaluation_data.dmas[dma]
                )

                if len(dma_specific_data[dma].flows.columns) == 0:
                    dma_specific_data[dma].flows[
                        "PUMP_1"
                    ] = simple_evaluation_data.flows[
                        self.hyperparameters["default_flow_sensor"]
                    ]
                else:
                    dma_specific_data[dma].flows["PUMP_1"] = dma_specific_data[
                        dma
                    ].flows.sum(axis=1)
        else:
            if self.hyperparameters["default_flow_sensor"] == "sum":
                simple_evaluation_data.flows[
                    "PUMP_1"
                ] = simple_evaluation_data.flows.sum(axis=1)
            else:
                simple_evaluation_data.flows["PUMP_1"] = simple_evaluation_data.flows[
                    self.hyperparameters["default_flow_sensor"]
                ]
            dma_specific_data["main"] = simple_evaluation_data

        if self.trained == False:
            # TODO: Implement reoccurring training on trailing timeframe?
            start_time = pd.to_datetime(self.hyperparameters["leakfree_time_start"])
            end_time = pd.to_datetime(self.hyperparameters["leakfree_time_stop"])
            for dma_key in dma_specific_data.keys():
                self._train(
                    dma_specific_data[dma_key], start_time, end_time, dma=dma_key
                )

        results = []
        for dma_key in dma_specific_data.keys():
            leaks = self._detect(dma_specific_data[dma_key], dma_key)

            for leak_pipe, leak_start in zip(leaks.index, leaks):
                results.append(
                    BenchmarkLeakageResult(
                        leak_pipe_id=leak_pipe,
                        leak_time_start=leak_start,
                        leak_time_end=leak_start,
                        leak_time_peak=leak_start,
                    )
                )
        return results

    def detect_online(self, evaluation_data) -> BenchmarkLeakageResult:
        scada_data = SCADA_data()

        scada_data.pressures = evaluation_data.pressures
        nodes = evaluation_data.pressures.keys()

        flows = evaluation_data.flows
        # TODO: Make algorithm independent of pump name
        # flows = flows.rename(columns={"P-01": "PUMP_1"})
        scada_data.flows = flows

        # Leak Analyiss Fuction
        # nodes - List of Nodes which should be.
        # scada - Dataset which holds flow and pressure data.
        # cor_time_frame - Time Frame for where there is no leak.
        # def leak_analysis(nodes, scada_data, cor_time_frame):
        N = len(nodes)
        # T = Timestamps
        T = scada_data.pressures.shape[0]
        P = scada_data.pressures[nodes].values
        V = scada_data.flows["PUMP_1"].values

        np.fill_diagonal(self.K0, 0)
        np.fill_diagonal(self.K1, 1)
        np.fill_diagonal(self.Kd, 0)

        e = (
            np.multiply.outer(self.K0, np.ones(T))
            + np.multiply.outer(self.Kd, V)
            + np.multiply.outer(self.K1, np.ones(T))
            * np.multiply.outer(P, np.ones(N)).transpose(1, 2, 0)
            - np.multiply.outer(P, np.ones(N)).transpose(2, 1, 0)
        )

        # # Why?
        # if 'n215' in nodes:
        #     e[nodes.index('n215'), :, :] *= 0.02

        # Faster:
        e_count = e.copy()
        e_count[e_count < 0] = 0
        e_count[e_count > 0] = 1
        # Find Sensor Indexes which are most affected
        # As the previous implementation used quicksort (with descending order) we have to use this beautiful hack
        max_affected_sensors_index = np.abs(
            (e_count.sum(axis=0)[::-1]).argsort(kind="quicksort", axis=0) - (N - 1)
        )[-1, :]
        # There are better ways, which need to be tested.
        # # 1. Use stable sort
        # max_affected_sensors_index = e_count.sum(
        #     axis=0).argsort(kind='stable', axis=0)[-1, :]

        # # 2. Simply use argmax
        # max_affected_sensors_index = e_count.sum(axis=0).argmax(axis=0)

        # Select Values of most affected sensors (n, T)
        max_affected_sensor_values = np.take_along_axis(
            e,
            np.expand_dims(np.expand_dims(max_affected_sensors_index, axis=0), axis=0),
            axis=1,
        ).squeeze()
        norm_values = np.linalg.norm(max_affected_sensor_values, axis=0)
        res = np.zeros((N, T))
        np.put_along_axis(
            res, np.expand_dims(max_affected_sensors_index, axis=0), norm_values, axis=0
        )

        MRE = pd.DataFrame(res.T, index=scada_data.pressures.index, columns=nodes)
        leaks, cusum_data = cusum(MRE)

        if self.debug:
            MRE.to_csv(self.additional_output_path + "mre.csv")
            # print(MRE)
            # for sensor in MRE.columns:
            #     MRE_single = MRE[[sensor]]
            #     CUSUM_DATA = cusum(MRE_single, est_length="1 minute")
            #     # print(sensor)
            #     print(CUSUM_DATA[0])
            # print(leaks)
            # print(rawdata)
            cusum_data.to_csv(self.additional_output_path + "cusum.csv")

        # Overall MRE is not good for detection, so we just keep these Nodes as Sensors to Consider in the next Step

        results = []
        for leak_pipe, leak_start in zip(leaks.index, leaks):
            results.append(
                BenchmarkLeakageResult(
                    leak_pipe_id=leak_pipe,
                    leak_time_start=leak_start,
                    leak_time_end=leak_start,
                    leak_time_peak=leak_start,
                )
            )
        return results


# algorithm = CustomAlgorithm()
# hyperparameters = [
#     Hyperparameter(*{
#         'name': 'K0',
#         'type': 'float',
#         'min': 0,
#         'max': 1,
#         'default': 0.5,
#         'step': 0.1,
#     }),
# ]


# TODO: Faster implementation: https://stackoverflow.com/questions/40954560/pandas-rolling-apply-custom
def zscore(df, win):
    """calcualte rolling z_score for leak trajectories
    df :    dfs containing leak trajectories (model reconstruction errors)
    win :   window for error statistics calculation

    df_z :  pd.DataFrame containing normalized trajectories
    """
    start_dates = df.ne(0).idxmax()
    z_base = np.zeros(shape=(win, df.shape[1]))

    for i, pipe in enumerate(df):
        start = start_dates[pipe]
        stop = start_dates[pipe] + pd.Timedelta(win, unit="Min") * 5
        z_base[:, i] = df[pipe].loc[start:stop].iloc[:-1]

    m = z_base.mean(axis=0)
    sigma = z_base.std(axis=0)
    z = (df - m) / sigma
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_z = pd.DataFrame(z, columns=df.columns, index=df.index)
    #
    return df_z, sigma, m
