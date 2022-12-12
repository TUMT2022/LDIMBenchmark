# In[1]:
import wntr
import wntr.network.controls as controls
import numpy as np
import matplotlib.pyplot as plt
import math


def generatePoulakisNetwork(network_size=6, max_pipes=0, max_junctions=0):
    """ "
    Generates a Poulakis network with (network_size,network_size) junctions.
    """

    if network_size == 0 and max_pipes == 0 and max_junctions == 0:
        raise ValueError("You have to set at least one parameter")
    if max_junctions < 3 and max_junctions != 0:
        raise ValueError("Max junctions must be greater than 2")
    # TODO Create warning if max_pipes or max_junctions interact with each oter
    # (e.g. because of the number of junctions being too small to generate more pipes)

    # Choose the network size automatically
    if network_size == 0 and max_pipes != 0:
        # Reverse of calculating the number of pipes from network_size
        # pipes = network_size * (network_size - 2) + (network_size-1) * (network_size-1) + 1
        network_size = math.ceil(math.sqrt(max_pipes) / math.sqrt(2) + 1)

    if network_size == 0 and max_junctions != 0:
        # Reverse of calculating the number of junctions from network_size
        # junctions = network_size * (network_size-1)+1
        network_size = math.ceil(1 / 2 * (math.sqrt(4 * max_junctions - 3) + 1))
    if network_size < 3:
        network_size = 3

    # If max_pipes is set to high
    # pipes = netwrok_size * (network_size - 1) + network_size * network_size

    wn = wntr.network.WaterNetworkModel()
    wn.options.graphics.dimensions = (0, 0, 10, 10)
    wn.options.hydraulic.trials = 40
    wn.options.hydraulic.accuracy = 10e-10
    wn.options.hydraulic.emitter_exponent = 1
    wn.units = "SI"
    wn.options.hydraulic.inpfile_units = "LPS"
    # wn.options.hydraulic.inpfile_pressure_units = 'm'

    # Because we use WNTRSimulator, we need to convert D-W to H-W
    wn.options.hydraulic.headloss = "H-W"
    # From Poulakis Paper (darcy-weißback) 0.25 mm == 0.0008530184 feet
    # Conversion via EPANET Manual p. 18 = 130-140 (H-W) / https://www.ewra.net/ew/pdf/EW_2017_58_74.pdf (Table 1)
    pipe_roughness = 127  # Hazen Williams Coefficient (unitless)

    # If EPANET is used these more accurate values can be used:
    # wn.options.hydraulic.headloss = 'D-W'
    # pipe_roughness = 0.26 # Dary Weißbach Roughness Coefficient - mm

    base_demand = 0.05  # m³/s

    # wn.add_tank("J-01", elevation=52, coordinates=(0, 1), diameter=1550*5, min_level=0, max_level=1, init_level=1, )
    # Simulating Tank with height of 52m but unlimited water supply
    wn.add_reservoir("J-01", base_head=52, coordinates=(0, 1))

    pipe_diameter_factor = 1 if network_size == 6 else network_size / 6 * 0.95

    def get_diameter(x, y):
        if x + y <= network_size - 3:
            return 0.600 * pipe_diameter_factor
        if x + y <= network_size - 1:
            return 0.450 * pipe_diameter_factor
        return 0.300 * pipe_diameter_factor

    horizonal_pipe_thickness = np.zeros([network_size - 2, network_size - 1])
    vertical_pipe_thickness = np.zeros([network_size - 2, network_size + 1])
    # x = 2
    for i in range(1, network_size):
        for j in range(1, network_size + 1):
            junction_number = (i - 1) * network_size + j + 1

            horizontal_pipe_number = (
                (i - 1) * (network_size - 1) + (i - 1) * network_size + j
            )
            vertical_pipe_number = (
                (i - 1) * (network_size - 1) + (i - 2) * network_size + 1 + j
            )

            if (
                horizontal_pipe_number > max_pipes
                and vertical_pipe_number > max_pipes
                and max_pipes != 0
            ):
                break
            if junction_number > max_junctions and max_junctions != 0:
                break
            current_junction = "J-{:02d}".format(junction_number)
            wn.add_junction(
                current_junction,
                base_demand=base_demand,
                demand_pattern="pat1",
                elevation=0,
                coordinates=(i * 2, -j),
            )

            # Add Vertical Pipes
            if i >= 2:
                if not (vertical_pipe_number > max_pipes and max_pipes != 0):
                    diameter = get_diameter(i, j)
                    # print(f"{p} {i} {j}: {diameter}")
                    vertical_pipe_thickness[i - 2, j - 1] = diameter
                    above_junction = "J-{:02d}".format(junction_number - network_size)
                    wn.add_pipe(
                        "P-{:02d}".format(vertical_pipe_number),
                        current_junction,
                        above_junction,
                        length=2000,
                        diameter=diameter,
                        roughness=pipe_roughness,
                    )

            # Add Horizontal Pipes
            if j >= 2:
                if not (horizontal_pipe_number > max_pipes and max_pipes != 0):
                    diameter = get_diameter(i, j)
                    # print(f"{p} {i} {j}: {diameter}")
                    horizonal_pipe_thickness[i - 2, j - 2] = diameter
                    above_junction = "J-{:02d}".format(junction_number - 1)
                    wn.add_pipe(
                        "P-{:02d}".format(horizontal_pipe_number),
                        current_junction,
                        above_junction,
                        length=1000,
                        diameter=diameter,
                        roughness=pipe_roughness,
                    )

    wn.add_pipe(
        "P-01",
        "J-01",
        "J-02",
        length=100,
        diameter=get_diameter(0, 0),
        roughness=pipe_roughness,
    )
    return wn


# wn = generatePoulakisNetwork(5)  # , max_pipes=9)
# fig, ax = plt.subplots(1, 1, figsize=(12, 10))
# ax = wntr.graphics.plot_network(wn, ax=ax, title="Poulakis Network",
#                                 node_labels=True, link_labels=True,)  # node_attribute='elevation',)
# print(wn.describe())


# # In[11]:
# network_size = 5
# # Pipes
# pipe = network_size * (network_size - 2) + \
#     (network_size-1) * (network_size-1) + 1
# # Junctions
# network_size * (network_size-1)+1


# y = 26
# math.ceil(math.sqrt(y)/math.sqrt(2) + 1)

# math.ceil(1/2*(math.sqrt(4*y - 3) + 1))
# %%
