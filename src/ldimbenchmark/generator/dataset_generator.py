import wntr
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import os
import yaml
from math import sqrt
import math
from pydantic import BaseModel
from typing import List, Union
from wntr.network import write_inpfile
from ldimbenchmark.classes import BenchmarkLeakageResult


class DatasetGeneratorConfigModel(BaseModel):
    startTime: datetime
    endTime: datetime
    timestep: str


class DatasetGeneratorConfigLeak(BaseModel):
    linkID: str
    startTime: datetime
    peakTime: datetime
    endTime: datetime
    leakDiameter: float  # (m)


class DatasetGeneratorConfig(BaseModel):
    pressure_sensors: Union[str, List[str]]
    flow_sensors: Union[str, List[str]]
    amrs: Union[str, List[str]]
    level_sensors: Union[str, List[str]]
    model: DatasetGeneratorConfigModel
    leakages: List[DatasetGeneratorConfigLeak]


class DatasetGenerator:
    def __init__(self, water_network_model, configVar):
        config = DatasetGeneratorConfig(**configVar)
        # TODO: Check that leaks are withing timeframe

        # Statics
        # demand-driven (DD) or pressure dependent demand (PDD)
        self.mode_simulation = "PDD"  # 'PDD'#'PDD'

        if config.pressure_sensors == "all":
            self.pressure_sensors = water_network_model.node_name_list
        else:
            self.pressure_sensors = config.pressure_sensors
        self.flow_sensors = config.flow_sensors
        self.amrs = config.amrs
        self.level_sensors = config.level_sensors

        # Initialization
        self.simulation_start_time = config.model.startTime
        self.simulation_end_time = config.model.endTime
        self.leakages = config.leakages
        timestep = pd.to_timedelta(config.model.timestep)
        self.time_step = round(timestep.total_seconds())
        self.time_stamps = pd.date_range(
            self.simulation_start_time, self.simulation_end_time, freq=timestep
        )
        self.time_stamps_count = len(self.time_stamps) - 1

        # Assign EPANET network
        self.wn = water_network_model
        # Set Model Simulatiom Parameters
        self.wn.options.hydraulic.demand_model = self.mode_simulation
        self.wn.options.time.hydraulic_timestep = self.time_step
        self.wn.options.time.report_timestep = self.time_step
        self.wn.options.time.pattern_timestep = self.time_step
        self.wn.options.time.duration = self.time_stamps_count * self.time_step
        # self.wn.options.report.
        # Why?
        for name, node in self.wn.junctions():
            node.required_pressure = 25
            # print(node.required_pressure)
            # print(node.minimum_pressure)

    def generate(self):
        wn_with_leaks = self.wn
        self.leaks = []
        for leak in self.leakages:
            # Split pipe and add a leak node

            leak_start_time = self.time_stamps.get_loc(leak.startTime)
            leak_end_time = self.time_stamps.get_loc(leak.endTime)
            leak_peak_time = self.time_stamps.get_loc(leak.peakTime)
            leak_diameter = float(leak.leakDiameter)

            # Classify as abrupt leak if maximum is reached within 24 hours
            leak_type = (
                "abrupt"
                if (leak_peak_time - leak_start_time) * self.time_step < 60 * 60 * 24
                else "incipient"
            )

            leak_area = math.pi * (leak_diameter / 2) ** 2

            # Split pipe to add a leak
            pipe_id = wn_with_leaks.get_link(leak.linkID)
            node_leak = f"{pipe_id}_leaknode"
            # Splits pipe and adds junction with zero-demand
            wn_with_leaks = wntr.morph.split_pipe(
                wn_with_leaks, pipe_id, f"{pipe_id}_B", node_leak
            )
            leak_node = wn_with_leaks.get_node(node_leak)

            # Generate linearly increasing leak diameter till peak time
            steps_till_peak = leak_peak_time - leak_start_time
            increment_leak_diameter = np.linspace(
                0, leak_diameter, num=steps_till_peak, endpoint=True
            )

            emitter_scale = 0.75 * sqrt(2 / 1000) * 990.27
            increment_leak_area = (
                emitter_scale * math.pi * (increment_leak_diameter / 2) ** 2 * 10000
            )

            leak_max_magnitude = increment_leak_area[len(increment_leak_area) - 1]

            pattern_array = (
                [0] * leak_start_time
                + increment_leak_area.tolist()
                + [leak_max_magnitude] * (leak_end_time - leak_peak_time)
                + [0] * (self.time_stamps_count - leak_end_time)
            )

            # plt.plot(pattern_array)

            # Remove default zero-demand
            del leak_node.demand_timeseries_list[0]
            # basedemand
            pattern_name = f"{str(leak_node)}_pattern"
            wn_with_leaks.add_pattern(pattern_name, pattern_array)
            leak_node.add_demand(1, pattern_name)

            leak_node.required_pressure = 1
            # Pressure below demand is not filled at all
            leak_node.minimum_pressure = 0

            # save times of leak
            leak_start = self.time_stamps[leak_start_time]
            leak_start_time_string = leak_start._date_repr + " " + leak_start._time_repr
            leak_end = self.time_stamps[leak_end_time]
            leak_end_time_string = leak_end._date_repr + " " + leak_end._time_repr
            leak_peak_time_string = (
                self.time_stamps[leak_peak_time]._date_repr
                + " "
                + self.time_stamps[leak_peak_time]._time_repr
            )

            leak_pipe_nodes = [pipe_id.start_node_name, pipe_id.end_node_name]

            self.leaks.append(
                BenchmarkLeakageResult(
                    leak_pipe_id=str(pipe_id),
                    leak_pipe_nodes=leak_pipe_nodes,
                    leak_node=str(leak_node),
                    leak_diameter=leak_diameter,
                    leak_area=leak_area,
                    leak_type=leak_type,
                    leak_time_start=leak_start_time_string,
                    leak_time_peak=leak_peak_time_string,
                    leak_time_end=leak_end_time_string,
                )
            )

        # Save the water network model to a file before using it in a simulation
        # with open("wn_with_leaks.pickle", "wb") as f:
        #     pickle.dump(wn_with_leaks, f)

        # Run wntr simulator

        # wntr.graphics.plot_network(wn_with_leaks, title="Poulakis Network", node_labels=True, link_labels=True,)
        # wn_with_leaks.write_inpfile("out/simulation/simulated.inp")
        # wn_with_leaks = wntr.network.read_inpfile("out/simulation/simulated.inp")
        sim = wntr.sim.WNTRSimulator(wn_with_leaks)

        self.results = sim.run_sim()
        if self.results.node["pressure"].empty:
            print("Negative pressures.")
            return -1

        for leak in self.leaks:
            leak["leak_max_flow"] = self.results.node["demand"][leak["leak_node"]].max()

        # TODO: Add total_leak_flow
        self.leak_dataframe = pd.DataFrame(self.leaks)
        return self.time_stamps, self.results, self.leak_dataframe, wn_with_leaks

    def write_generated_data(self, results_folder, model_name="synthetic_dataset"):
        if self.results == None:
            print("Run the 'dataset_generator()' Function before. No results to write.")
            return

        # Create CSV files
        decimal_size = 4

        training_evaluation_split = (
            self.simulation_start_time
            + (self.leakages[0].startTime - self.simulation_start_time) / 2
        )
        # TODO: Fix start and end times to be exclusive, should be (instead of same day for training and evaluation)
        #     start: 2019-01-01 00:00
        #     end: 2019-12-31 23:55
        #   training:
        #     start: 2018-01-01 00:00
        #     end: 2019-12-31 23:55
        dataset_info = f"""
        name: {model_name}
        inp_file: model.inp
        dataset:
          training:
            start: '{str(self.simulation_start_time)}'
            end: '{str(training_evaluation_split)}'
          evaluation:
            start: '{str(training_evaluation_split)}'
            end: '{str(self.simulation_end_time)}'
        """
        # Convert info to yaml dictionary
        dataset_info = yaml.safe_load(dataset_info)
        dataset_info["leakages"] = self.leak_dataframe.to_dict("records")
        self.leak_dataframe.to_csv(os.path.join(results_folder, "leaks.csv"))

        # Write info to file
        with open(os.path.join(results_folder, f"dataset_info.yaml"), "w") as f:
            yaml.dump(dataset_info, f)

        write_inpfile(self.wn, os.path.join(results_folder, f"model.inp"))

        # Map Index to TimeStamp
        leak_values = pd.DataFrame(index=self.time_stamps)
        # leak_values.index = self.time_stamps
        for index, leak in self.leak_dataframe.iterrows():
            leak_values[leak["leak_pipe_id"]] = self.results.node["demand"][
                leak["leak_node"]
            ].values[: len(self.time_stamps)]

        leak_values.round(decimal_size).to_csv(
            os.path.join(results_folder, f"leakages_demand.csv"),
            index_label="Timestamp",
        )

        # Create xlsx file with Measurements
        total_pressures = pd.DataFrame(index=self.time_stamps)
        total_demands = pd.DataFrame(index=self.time_stamps)
        total_flows = pd.DataFrame(index=self.time_stamps)
        total_levels = pd.DataFrame(index=self.time_stamps)

        for folders in ["pressures", "demands", "flows", "levels"]:
            os.makedirs(os.path.join(results_folder, folders), exist_ok=True)

        for node_id, values in self.results.node["demand"].items():
            if (
                node_id in self.pressure_sensors
                and self.wn.get_node(node_id).node_type == "Junction"
            ):
                pres = self.results.node["pressure"][node_id].values[
                    : len(self.time_stamps)
                ]
                pd.DataFrame(pres, index=self.time_stamps).round(decimal_size).to_csv(
                    os.path.join(results_folder, "pressures", f"{node_id}.csv"),
                    index_label="Timestamp",
                    header=[node_id],
                )
                # pres = pres[:len(self.time_stamp)]
                total_pressures[node_id] = pres

            if node_id in self.amrs:
                dem = self.results.node["demand"][node_id].values[
                    : len(self.time_stamps)
                ]
                pd.DataFrame(dem, index=self.time_stamps).round(decimal_size).to_csv(
                    os.path.join(results_folder, "demands", f"{node_id}.csv"),
                    index_label="Timestamp",
                    header=[node_id],
                )
                # dem = dem[:len(self.time_stamp)]
                # dem = [elem * 3600 * 1000 for elem in dem] #CMH / L/s
                total_demands[node_id] = dem

            if node_id in self.level_sensors:
                level_pres = self.results.node["pressure"][node_id].values[
                    : len(self.time_stamps)
                ]
                pd.DataFrame(level_pres, index=self.time_stamps).round(
                    decimal_size
                ).to_csv(
                    os.path.join(results_folder, "levels", f"{node_id}.csv"),
                    index_label="Timestamp",
                    header=[node_id],
                )
                # level_pres = level_pres[:len(self.time_stamp)]
                # level_pres = [round(elem, decimal_size) for elem in level_pres]
                total_levels[node_id] = level_pres

        for link_id, values in self.results.link["flowrate"].items():
            if link_id in self.flow_sensors:
                flows = self.results.link["flowrate"][link_id].values[
                    : len(self.time_stamps)
                ]
                pd.DataFrame(flows, index=self.time_stamps).round(decimal_size).to_csv(
                    os.path.join(results_folder, "flows", f"{link_id}.csv"),
                    index_label="Timestamp",
                    header=[link_id],
                )
                total_flows[link_id] = flows

        # Pressures (m), Demands (m^3/s), Flows (m^3/s), Levels (m)
        # total_pressures.round(decimal_size).to_csv(
        #     os.path.join(results_folder, "pressures.csv"), index_label="Timestamp"
        # )
        # total_demands.round(decimal_size).to_csv(
        #     os.path.join(results_folder, "demands.csv"), index_label="Timestamp"
        # )
        # total_flows.round(decimal_size).to_csv(
        #     os.path.join(results_folder, "flows.csv"), index_label="Timestamp"
        # )
        # total_levels.round(decimal_size).to_csv(
        #     os.path.join(results_folder, "levels.csv"), index_label="Timestamp"
        # )
