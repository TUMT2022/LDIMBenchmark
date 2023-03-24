# %%
%load_ext autoreload
%autoreload 2
from ldimbenchmark.datasets.analysis import DatasetAnalyzer
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark.datasets.derivation import DatasetDerivator
from ldimbenchmark.generator import generateDatasetForTimeSpanDays
from ldimbenchmark.methods import MNF, LILA

from ldimbenchmark.benchmark import LDIMBenchmark
import logging
import os
from matplotlib import pyplot as plt

test_data_folder = "test_data"

test_data_folder_datasets = os.path.join("test_data", "datasets")

#%%

# generated_dataset_path = os.path.join(test_data_folder_datasets, "graz-ragnitz")
# generated_dataset_path = os.path.join(test_data_folder_datasets, "battledim")
generated_dataset_path = os.path.join(test_data_folder_datasets, "gjovik")
dataset = Dataset(generated_dataset_path)


dataset.loadData()

# Plot Overview of timeseries sensors

# fig = plt.figure(figsize=(40, 50))

# gs = fig.add_gridspec(4, hspace=0, height_ratios=[len(getattr(dataset, data_name)) for data_name in ["demands", "pressures", "flows", "levels"]])
# axes = gs.subplots(
#     sharex=True, sharey=False
# )

for index, data_name in enumerate(["pressures", "flows", "levels"]): #"demands",
    data_group = getattr(dataset, data_name)
    offset = 0
    for sensor_name, sensor_data in data_group.items():
        # create a horizontal plot
        colors = ["C{}".format(i) for i in range(len(data_group))]

        axes[index].eventplot(sensor_data.index, lineoffsets=offset, linelengths=0.8, label=sensor_name) #, colors=colors)
        offset = offset + 1
        sensor_data
        break
    break
    axes[index].set_yticks(range(len(data_group)))
    axes[index].set_yticklabels(data_group.keys())
    axes[index].set_title(data_name, y=1.0, pad=-14)
    # axes[index].axvline(dataset.info["dataset"]["evaluation"]["start"])
    # x_ticks=np.array(sensor_data.index)

    # x_ticks_1=pd.date_range(start=x_ticks.min(), end=x_ticks.max())
    # axs.set_xticklabels(x_ticks_1,rotation = 45)

# plt.show()
# fig.savefig(f"out/{dataset.id}.png")
# plt.close(fig)

sensor_data["date"] = sensor_data.index


sensor_data['Group_id'] = (
    sensor_data["date"]
      .diff()
      .dt
      .seconds
      .div((240))
      .gt(1)
      .cumsum()
)
def first_last(df):
    return df.ix[[0, -1]]


test = sensor_data.groupby("Group_id").nth([0,-1])

test


# %%
import pandas as pd

df['Group_id'] = (
    
)
df

#%%


analysis = DatasetAnalyzer(os.path.join(test_data_folder, "out"))

analysis.analyze(dataset)