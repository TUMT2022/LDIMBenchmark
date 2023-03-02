from ldimbenchmark import (
    BenchmarkData,
    BenchmarkLeakageResult,
)
from ldimbenchmark.classes import (
    Hyperparameter,
    LDIMMethodBase,
    MethodMetadata,
    MethodMetadataDataNeeded,
)
from ldimbenchmark.methods.utils.cusum import cusum

import pickle
import math
import os
from os import path
import tempfile
import wntr
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
import copy

from ldimbenchmark.utilities import simplifyBenchmarkData


class DUALMethod(LDIMMethodBase):
    def __init__(self):
        super().__init__(
            name="dualmethod",
            version="0.1.0",
            metadata=MethodMetadata(
                data_needed=MethodMetadataDataNeeded(
                    pressures="necessary",
                    flows="necessary",
                    levels="optional",
                    model="ignored",
                    demands="ignored",
                    structure="necessary",
                ),
                hyperparameters=[
                    Hyperparameter(
                        name="est_length",
                        description="Length of the estimation period in hours",
                        value_type=int,
                        default=20 * 24,  # 20 days
                        min=1,
                        max=8760,  # 1 year
                    ),
                    Hyperparameter(
                        name="C_threshold",
                        description="Threshold for the CUSUM statistic",
                        value_type=float,
                        default=0.2,
                        max=10.0,
                        min=0.0,
                    ),
                    Hyperparameter(
                        name="delta",
                        description="Delta for the CUSUM statistic",
                        value_type=float,
                        default=0.3,
                        max=10.0,
                        min=0.0,
                    ),
                ],
            ),
        )

    def init_with_benchmark_params(
        self, additional_output_path=None, hyperparameters={}
    ):
        super().init_with_benchmark_params(additional_output_path, hyperparameters)
        if additional_output_path is not None:
            self.path_to_model_pickle = path.join(
                additional_output_path, "dualmodel.pickle"
            )

    def prepare(self, train_data: BenchmarkData = None):
        # TODO: Calibrate Model (for now just use the model given)

        # TODO: Calibrate roughness values of the pipes
        if False:
            # DNHB: Roughness coefficients found in Lippacher (2018) for each pipe group?
            g1 = [0.0, 0.0809]
            g2 = [0.0809001, 0.089]
            g3 = [0.0890001, 0.0999]
            g4 = 0.09990001

            r1 = 0.005  # 0.9279314753976028
            r2 = 0.005  # 0.07775810227857316
            r3 = 0.005  # 0.010012824359365236
            r4 = 0.005  # 0.3838801559425374

            for pipename in wn.link_name_list:
                pipe = wn.get_link(pipename)

                if pipe.link_type == "Pipe":
                    diameter = pipe.diameter

                    if g1[0] < diameter < g1[1]:
                        pipe.roughness = r1
                    elif g2[0] < diameter < g2[1]:
                        pipe.roughness = r2
                    elif g3[0] < diameter < g3[1]:
                        pipe.roughness = r3
                    else:
                        pipe.roughness = r4

        # TODO: Scaling demand multiplier
        # p = wn.get_pattern('1')
        # p.multipliers = 0.85

        # TODO: Refine the model with the training data...

    def detect_offline(
        self, evaluation_data: BenchmarkData
    ) -> List[BenchmarkLeakageResult]:
        simple_evaluation_data = simplifyBenchmarkData(
            evaluation_data, resample_frequency="5T"
        )

        # Custom Deepcopy
        # self.wn = copy.deepcopy(train_data.model)
        temp_dir = tempfile.TemporaryDirectory()
        path_to_model_pickle = path.join(temp_dir.name, "dualmodel.pickle")
        with open(path_to_model_pickle, "wb") as f:
            pickle.dump(evaluation_data.model, f)

        with open(path_to_model_pickle, "rb") as f:
            self.wn = pickle.load(f)
        temp_dir.cleanup()

        pressure_sensors_with_data = simple_evaluation_data.pressures.keys()
        pipelist = list(
            filter(
                lambda link: self.wn.get_link(link).link_type == "Pipe",
                self.wn.link_name_list,
            )
        )
        start = simple_evaluation_data.pressures.index[0]
        end = simple_evaluation_data.pressures.index[-1]
        duration = end - start
        frequency = (
            simple_evaluation_data.pressures.index[1]
            - simple_evaluation_data.pressures.index[0]
        )

        ###
        # 1. Step: Build the Dual model
        ###
        pressure_sensors_with_data = simple_evaluation_data.pressures.keys()

        for sensor in pressure_sensors_with_data:
            node = self.wn.get_node(sensor)

            elevation = node.elevation
            coordinates = node.coordinates
            pattern_name = f"pressurepattern_{sensor}"

            # Question: Why addition with elevation?
            # Reservoirs do not specify the eleveation so we have to add it to the pressure pattern
            # The head of the reservoir for each time step equals the measured pressure plus the node elevation.
            # This shifts the boundary condition from the fixed-demand at the sensor nodes, to the fixed-head at the
            # corresponding virtual reservoir.
            # As a result, the previous boundary conditionnode becomes a free variable available for modelled input
            self.wn.add_pattern(
                name=pattern_name,
                pattern=list(
                    simple_evaluation_data.pressures[sensor].values + elevation
                ),
            )

            self.wn.add_reservoir(
                name=f"dualmodel_reservoir_{sensor}",
                base_head=1.0,
                head_pattern=pattern_name,
                coordinates=coordinates,
            )

            self.wn.add_junction(
                name=f"dualmodel_node_{sensor}",
                coordinates=coordinates,
                elevation=elevation,
            )

            # TODO: Roughness has to be set per model (HW/DW etc.)
            self.wn.add_pipe(
                name=f"dualmodel_{sensor}",
                start_node_name=f"dualmodel_node_{sensor}",
                end_node_name=f"dualmodel_reservoir_{sensor}",
                check_valve=False,
                diameter=0.1,
                length=1,
                roughness=127,
            )

            self.wn.add_valve(
                name=f"dualmodel_valve_{sensor}",
                start_node_name=f"dualmodel_node_{sensor}",
                end_node_name=f"{sensor}",
                valve_type="TCV",
                diameter=0.1,
                initial_setting=1.0,
            )  # TCV = Throttle Control Valve, initial_setting controls the loss coefficient (0-1)

        # TODO: Incorporate Flows into model
        # TODO: Also incorporate Tank Levels to the model
        # Set patterns for reservoirs from measurements
        # for reservoir in simple_evaluation_data.levels.keys():
        #     res = wn.get_node(reservoir)
        #     base_head = res.base_head
        #     wn.add_pattern(name=f'reservoirhead_{res_name}', pattern=list(
        #         ((level + base_head) / base_head).values))
        #     res.head_pattern_name = f'reservoirhead_{res_name}'

        # Report and simulation settings
        self.wn.options.time.duration = int(duration.total_seconds())
        self.wn.options.time.hydraulic_timestep = int(
            pd.to_timedelta(frequency).total_seconds()
        )
        self.wn.options.time.pattern_timestep = int(
            pd.to_timedelta(frequency).total_seconds()
        )
        self.wn.options.time.report_timestep = int(
            pd.to_timedelta(frequency).total_seconds()
        )
        self.wn.options.time.rule_timestep = int(
            pd.to_timedelta(frequency).total_seconds()
        )

        # TODO: second step is not needed for detection
        # We can simulate once and use the virtual reservoirs directly for the detection of leaks

        ############################################################
        # 2. Step: Run the simulation for each pipe
        ############################################################
        tot_outflow = {}

        # path = os.path.join(
        #     self.additional_output_path, "tot_outflow_simulations", "test.pickle"
        # )
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        # if os.path.exists(path):
        #     with open(path, "rb") as f:
        #         result = pickle.load(f)
        # else:
        temp_dir = tempfile.TemporaryDirectory()
        sim = wntr.sim.EpanetSimulator(self.wn)
        # TODO: With multiprocessing this line produces a deadlock
        result = sim.run_sim(file_prefix=os.path.join(temp_dir.name, "all"))
        temp_dir.cleanup()
        # if not os.path.exists(path):
        #     with open(path, "wb") as f:
        #         pickle.dump(result, f)

        # Get the flow rate to the previously created extra reservoirs
        # in m³/s
        dualmodel_nodes = [
            "dualmodel_" + sensor for sensor in pressure_sensors_with_data
        ]
        leakflow = result.link["flowrate"][dualmodel_nodes].abs()
        # leakflow.index = simple_evaluation_data.pressures.index
        squareflow = leakflow  # **2
        tot_outflow = leakflow
        # tot_outflow = squareflow.sum(axis=1)  # sum per timestamp

        tot_outflow = pd.DataFrame(tot_outflow)
        tot_outflow.index = simple_evaluation_data.pressures.index

        if self.debug:
            tot_outflow.to_csv(
                os.path.join(self.additional_output_path, "tot_outflow.csv")
            )

            # import matplotlib as mpl
            # mpl.rcParams.update(mpl.rcParamsDefault)
            # plot = tot_outflow.plot()
            # fig = plot.get_figure()
            # fig.savefig(self.additional_output_path + "tot_outflow.png")

            # plot = simple_evaluation_data.pressures.plot()
            # fig = plot.get_figure()
            # fig.savefig(self.additional_output_path + "pressures.png")

        df_max = tot_outflow.abs()

        # Bring Data into the right format (only one value>0 per row)
        col_max = df_max.max(axis=1)
        mask = df_max.eq(col_max, axis=0)
        df_max = df_max.where(mask, other=0)
        # df_max.columns = ["all"]
        leaks, cusum_data = cusum(
            df_max,
            est_length=self.hyperparameters["est_length"],
            C_thr=self.hyperparameters["C_threshold"],
            delta=self.hyperparameters["delta"],
        )

        if self.debug:
            plot = col_max.plot()
            fig = plot.get_figure()
            fig.savefig(self.additional_output_path + "max.png")

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

        # results = pd.DataFrame(results)

        return results

    def detect_online(self, evaluation_data) -> BenchmarkLeakageResult:
        return None


algorithm = DUALMethod()
