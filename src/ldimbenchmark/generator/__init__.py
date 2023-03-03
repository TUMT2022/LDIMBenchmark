from concurrent.futures import ProcessPoolExecutor, as_completed
from ldimbenchmark.constants import LDIM_BENCHMARK_CACHE_DIR
from ldimbenchmark.generator.dataset_generator import (
    DatasetGenerator,
)
from ldimbenchmark.generator.poulakis_network import generatePoulakisNetwork
import wntr
import matplotlib.pyplot as plt
import os
import yaml
import numpy as np


import logging


# TODO: Move to main __init__.py
# parser = ArgumentParser()
# parser.add_argument(
#     "-o",
#     "--outputFolder",
#     dest="outputFolder",
#     default="./out",
#     help="the folder to generate the datasets in",
# )
# parser.add_argument(
#     "-c",
#     "--configurationFile",
#     dest="configurationFile",
#     default="dataset_configuration_evaluation.yml",
#     help="configuration file which will be used to generate the according dataset",
# )
# parser.add_argument(
#     "-n",
#     "--waterNetwork",
#     dest="waterNetwork",
#     default=None,
#     help="water network (.inp) file used to generate the dataset",
# )
# parser.add_argument(
#     "-v",
#     "--variations",
#     dest="variations",
#     default=None,
#     choices=["time", "junctions"],
#     help="ignore certain configurations and generate all variations of the property",
# )

# args = parser.parse_args()


# In[2]:

# Read input arguments from yalm file
# configuration_file_path = os.path.join(args.configurationFile)
# try:
#     with open(configuration_file_path, "r") as f:
#         config = yaml.safe_load(f.read())
# except:
#     print(f'"dataset_configuration" at {configuration_file_path} file not found.')
# sys.exit()

# yaml_example = """
# model:
#   startTime: 2022-01-01 00:00
#   endTime: 2022-03-01 00:00
#   timestep: 5min

# leakages:
#   - linkID: P-03
#     startTime: 2022-02-01 00:00
#     peakTime: 2022-02-15 12:00
#     endTime: 2022-03-01 00:00
#     leakDiameter: 0.011843  # (m)

# pressure_sensors: 'all'

# flow_sensors:
# - P-01

# level_sensors: []

# amrs:
# - J-03

# """
# config = yaml.safe_load(yaml_example)

# Just check if it is already valid
# DatasetGeneratorConfig(**config)
# water_network_model = wntr.network.WaterNetworkModel(
#     config['model']['filename'])

# In[3]:
# {'Nodes': 57, 'Links': 98
# for size in range(3, 8):
#     wn = generatePoulakisNetwork(size)
#     wn.write_inpfile(os.path.join(
#         results_folder, f"poulakis-{size}.inp"))
#     print(wn.describe())
#     fig, ax = plt.subplots(1, 1, figsize=(12, 10))
#     ax = wntr.graphics.plot_network(wn, ax=ax, title="Poulakis Network",
#                                     node_labels=True, link_labels=True,)  # node_attribute='elevation',)
#     fig.savefig(f"out/network_poulakis-{size}.png")


def generateDatasetForJunctionNumber(
    junctions: int, out_dir: str = LDIM_BENCHMARK_CACHE_DIR
):
    if os.path.exists(out_dir):
        logging.info(f"Skipping {out_dir} as it already exists")
        return
    os.makedirs(out_dir, exist_ok=True)

    wn = generatePoulakisNetwork(network_size=0, max_junctions=junctions)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax = wntr.graphics.plot_network(
        wn,
        ax=ax,
        title="Poulakis Network",
        node_labels=True,
        link_labels=True,
    )  # node_attribute='elevation',)
    fig.savefig(os.path.join(out_dir, f"network_poulakis-j-{junctions}.png"))

    yaml_example = """
    model:
      startTime: 2022-01-01 00:00
      endTime: 2022-03-01 00:00
      timestep: 5min

    leakages:
    - linkID: P-03
      startTime: 2022-02-01 00:00
      peakTime: 2022-02-15 12:00
      endTime: 2022-03-01 00:00
      leakDiameter: 0.011843  # (m)

    pressure_sensors: 'all'

    flow_sensors:
    - P-01

    level_sensors: []

    amrs:
    - J-03

    """
    config = yaml.safe_load(yaml_example)
    # Call leak dataset creator
    generator = DatasetGenerator(wn, config)
    generator.generate()
    generator.write_generated_data(out_dir, f"synthetic-j-{junctions}")


def generateDatasetForTimeSpanDays(days: int, out_dir: str):
    if os.path.exists(out_dir):
        logging.info(f"Skipping {out_dir} as it already exists")
        return
    os.makedirs(out_dir, exist_ok=True)
    wn = generatePoulakisNetwork()

    startDate = np.datetime64("2022-01-01 00:00")
    endDate = startDate + np.timedelta64(days, "D")
    yaml_example = """
    model:
      startTime: 2022-01-01 00:00
      endTime: 2022-03-01 00:00
      timestep: 5min

    leakages:
    - linkID: P-03
      startTime: 2022-02-01 00:00
      peakTime: 2022-02-15 12:00
      endTime: 2022-03-01 00:00
      leakDiameter: 0.011843  # (m)

    pressure_sensors: 'all'

    flow_sensors:
    - P-01

    level_sensors: []

    amrs:
    - J-03

    """
    config = yaml.safe_load(yaml_example)
    config["model"]["startTime"] = str(startDate)
    config["model"]["endTime"] = str(endDate)

    leakfree_timespan = int((days * 24) / 2)
    config["leakages"][0]["startTime"] = str(
        startDate + np.timedelta64(leakfree_timespan, "h")
    )
    config["leakages"][0]["peakTime"] = str(
        startDate + np.timedelta64(leakfree_timespan + 1, "h")
    )
    config["leakages"][0]["endTime"] = str(endDate)

    # Call leak dataset creator
    generator = DatasetGenerator(wn, config)
    generator.generate()
    generator.write_generated_data(out_dir, f"synthetic-days-{days}")


def generateDatasetsForJunctions(
    junction_count_low: int,
    junction_count_high: int,
    out_dir: str = LDIM_BENCHMARK_CACHE_DIR,
):
    parallel = True
    if parallel == True:
        junctions = range(junction_count_low, junction_count_high)
        arguments_list = zip(
            junctions,
            [
                os.path.join(out_dir, f"synthetic-j-{junction}")
                for junction in junctions
            ],
        )

        try:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                # submit all tasks and get future objects
                futures = [
                    executor.submit(generateDatasetForJunctionNumber, junction, num)
                    for junction, num in arguments_list
                ]
                # process results from tasks in order of task completion
                for future in as_completed(futures):
                    future.result()
        except KeyboardInterrupt:
            executor._processes.clear()
            os.kill(os.getpid(), 9)


def generateDatasetsForTimespan(
    days_low: int, days_high: int, out_dir: str = LDIM_BENCHMARK_CACHE_DIR
):
    parallel = True
    if parallel == True:
        days = range(days_low, days_high)
        arguments_list = zip(
            days, [os.path.join(out_dir, f"synthetic-days-{day}") for day in days]
        )

        try:
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                # submit all tasks and get future objects
                futures = [
                    executor.submit(generateDatasetForTimeSpanDays, day, num)
                    for day, num in arguments_list
                ]
                # process results from tasks in order of task completion
                for future in as_completed(futures):
                    future.result()
        except KeyboardInterrupt:
            executor._processes.clear()
            os.kill(os.getpid(), 9)


# if args.variations == "junctions":
#     generateDatasetsForJunctions(4, 59)

# for pipes in range(3, 57):
#     wn = generatePoulakisNetwork(8, max_pipes=pipes)
#     wn.write_inpfile(os.path.join(
#         results_folder, f"poulakis-p-{pipes}.inp"))
#     print(wn.describe())
#     fig, ax = plt.subplots(1, 1, figsize=(12, 10))
#     ax = wntr.graphics.plot_network(wn, ax=ax, title="Poulakis Network",
#                                     node_labels=True, link_labels=True,)  # node_attribute='elevation',)
#     fig.savefig(f"out/network_poulakis-p-{pipes}.png")

# In[11]:

# Generate Dataset for increasing timespan


# elif args.variations == "time":
#     generateDatasetsForTimespan(1, 61)


# Otherwise (if args.variations == None)
# else:
#     results_folder = args.outputFolder
#     os.makedirs(results_folder, exist_ok=True)

#     if args.waterNetwork is None:
#         wn = generatePoulakisNetwork()
#     else:
#         wn = wntr.network.WaterNetworkModel(args.waterNetwork)

#     fig, ax = plt.subplots(1, 1, figsize=(12, 10))
#     ax = wntr.graphics.plot_network(
#         wn,
#         ax=ax,
#         title="Poulakis Network",
#         node_labels=True,
#         link_labels=True,
#     )  # node_attribute='elevation',)
#     # fig.savefig(os.path.join(results_folder, "model.png"))

#     # Call leak dataset creator
#     generator = DatasetGenerator(wn, config)
#     # Create scenario one-by-one
#     generator.generate()

#     # wntr.graphics.plot_network(leak_wn, node_attribute=results.node['pressure'].loc[8000*300, :], link_attribute=results.link['flowrate'].loc[8000*300, :].abs(
#     # ), node_size=100, node_colorbar_label='Pressure', link_colorbar_label="Flowrate")
#     generator.write_generated_data(results_folder)


# TODO: make generator work as CLI

# model:
#   startTime: 2022-01-01 00:00
#   endTime: 2022-03-01 00:00
#   timestep: 5min

# leakages:
#   - linkID: P-03
#     startTime: 2022-02-01 00:00
#     peakTime: 2022-02-15 12:00
#     endTime: 2022-03-01 00:00
#     leakDiameter: 0.011843  # (m)

# pressure_sensors: 'all'

# flow_sensors:
# - P-01

# level_sensors: []

# amrs:
# - J-03

# python 2-dataset-generator/generate_synthetic_datasets.py --outputFolder 'datasets/synthetic/' --configurationFile '2-dataset-generator/dataset_configuration.yml'
