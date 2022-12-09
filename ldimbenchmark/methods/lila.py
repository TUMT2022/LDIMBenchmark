from ldimbenchmark.classes import LDIMMethodBase, MethodMetadata, Hyperparameter
from ldimbenchmark import (
    BenchmarkData,
    BenchmarkLeakageResult,
)

from datetime import timedelta
from sklearn.linear_model import LinearRegression
import sklearn
import pickle
import math

import numpy as np
import pandas as pd


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


def cusum(df, direction="p", delta=4, C_thr=3, est_length="3 days"):
    """Tabular CUSUM per Montgomery,D. 1996 "Introduction to Statistical Process Control" p318
       https://www.amazon.com/dp/0470169923
    df        :  data to analyze
    direction :  negative or positive? 'n' or 'p'
    delta     :  parameter to calculate slack value K =>  K = (delta/2)*sigma
    K         :  reference value, allowance, slack value for each pipe
    C_thr     :  threshold for raising flag
    est_length:  Window for estimating distribution parameters mu and sigma, needs to be longer than one timestep
    leak_det  :  Leaks detected
    """
    if est_length == "auto":
        distribution_window = pd.Timedelta(df.index[1] - df.index[0])
    else:
        distribution_window = pd.Timedelta(est_length)

    if df.index[1] - df.index[0] > distribution_window:
        raise ValueError("est_length needs to be longer than one timestep")

    ar_mean = np.zeros(df.shape[1])
    ar_sigma = np.zeros(df.shape[1])
    for i, col in enumerate(df.columns):
        traj_ = df[col].copy()
        traj_ = traj_.replace(0, np.nan).dropna()
        if len(traj_) == 0:
            ar_mean[i] = 0
            ar_sigma[i] = 0
            continue
        ar_mean[i] = traj_.loc[: (traj_.index[0] + distribution_window)].mean()
        ar_sigma[i] = traj_.loc[: (traj_.index[0] + distribution_window)].std()

    ar_K = (delta / 2) * ar_sigma

    cumsum = np.zeros(df.shape)

    if direction == "p":
        for i in range(1, df.shape[0]):
            cumsum[i, :] = [
                max(0, j) for j in df.iloc[i, :] - ar_mean + cumsum[i - 1, :] - ar_K
            ]
    elif direction == "n":
        for i in range(1, df.shape[0]):
            cumsum[i] = [
                max(0, j) for j in -df.iloc[i, :] + ar_mean + cumsum[i - 1, :] - ar_K
            ]

    # df_cs     :  pd.DataFrame containing cusum-values for each df column
    df_cs = pd.DataFrame(cumsum, columns=df.columns, index=df.index)

    # Shortcut:
    # df_cs = np.cumsum(df.replace(0, np.nan).dropna())
    # ar_sigma = df.std()
    leak_det = pd.Series(dtype=object)
    for i, pipe in enumerate(df_cs):
        C_thr_abs = C_thr * ar_sigma[i]
        if any(df_cs[pipe] > C_thr_abs):
            leak_det[pipe] = df_cs.index[(df_cs[pipe] > C_thr_abs).values][0]

    return leak_det, df_cs


class LILA(LDIMMethodBase):
    def __init__(self):
        super().__init__(
            name="LILA",
            version="0.1.0",
            metadata=MethodMetadata(
                data_needed=["pressures", "flows"],
                hyperparameters=[
                    Hyperparameter(
                        name="est_length",
                        description="Length of the estimation period",
                        default="3 days",
                        type=str,
                    ),
                    Hyperparameter(
                        name="C_threshold",
                        description="Threshold for the CUSUM statistic",
                        default=3,
                        type=int,
                    ),
                    Hyperparameter(
                        name="delta",
                        description="Delta for the CUSUM statistic",
                        default=4,
                        type=int,
                    ),
                ],
            ),
        )

    # TODO: Add DMA specific implementation (and hyperparameters)

    def train(self, train_data: BenchmarkData):
        # TODO: implement sensor level data loading
        # Load Data
        scada_data = SCADA_data()

        scada_data.pressures = train_data.pressures
        nodes = train_data.pressures.keys()

        flows = train_data.flows
        flows = flows.rename(columns={"P-01": "PUMP_1", "Inflow [l/s]": "PUMP_1"})
        scada_data.flows = flows

        N = len(nodes)

        self.K0 = np.zeros((N, N))
        self.K1 = np.zeros((N, N))
        self.Kd = np.zeros((N, N))

        for i, node in enumerate(nodes):
            ref_node = Ref_node(node)
            models = []

            X_tr = np.concatenate(
                [
                    scada_data.pressures[node].to_numpy().reshape(-1, 1),
                    scada_data.flows["PUMP_1"].to_numpy().reshape(-1, 1),
                ],
                axis=1,
            )

            for j, node_cor in enumerate(nodes):
                y_tr = scada_data.pressures[node_cor].to_numpy().reshape(-1, 1)
                model = LinearRegression()
                models.append(model.fit(X_tr, y_tr))
                self.K0[i, j] = model.intercept_[0]
                self.K1[i, j] = model.coef_[0][0]
                self.Kd[i, j] = model.coef_[0][1]

            ref_node.set_models(models)

    def detect(self, evaluation_data: BenchmarkData) -> list[BenchmarkLeakageResult]:
        scada_data = SCADA_data()

        scada_data.pressures = evaluation_data.pressures
        nodes = evaluation_data.pressures.keys()

        flows = evaluation_data.flows
        # TODO: Make algorithm independent of pump name
        flows = flows.rename(columns={"P-01": "PUMP_1", "Inflow [l/s]": "PUMP_1"})
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
        leaks, cusum_data = cusum(
            MRE,
            C_thr=self.hyperparameters["C_threshold"],
            delta=self.hyperparameters["delta"],
            est_length=self.hyperparameters["est_length"],
        )

        if self.logging:
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
                {
                    "pipe_id": leak_pipe,
                    "leak_start": leak_start,
                    "leak_end": leak_start,
                    "leak_peak": leak_start,
                }
            )
        return results

    def detect_datapoint(self, evaluation_data) -> BenchmarkLeakageResult:
        scada_data = SCADA_data()

        scada_data.pressures = evaluation_data.pressures
        nodes = evaluation_data.pressures.keys()

        flows = evaluation_data.flows
        # TODO: Make algorithm independent of pump name
        flows = flows.rename(columns={"P-01": "PUMP_1"})
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

        if self.logging:
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
                {
                    "pipe_id": leak_pipe,
                    "leak_start": leak_start,
                    "leak_end": leak_start,
                    "leak_peak": leak_start,
                }
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
