# %%

import math
import os
import re
import subprocess
import numpy as np

import pandas as pd
import big_o

# Calculate the size of the generated datasets


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(["du", "-s", path]).split()[0].decode("utf-8")


sizes = []
ns = []

folders = ["synthetic-days", "synthetic-junctions"]

for folder in folders:
    base_path = os.path.join(".ldim_benchmark_cache", "datagen", folder)
    for dir in sorted_alphanumeric(os.listdir(base_path)):
        print(dir)
        sizes.append(int(du(os.path.join(base_path, dir))))
        ns.append(int(re.split("([0-9]+)", dir)[1]))

    values = pd.DataFrame(sizes, index=ns)

    bigo_result = big_o.infer_big_o_class(
        np.array(values.index), np.array(values[0]), verbose=True
    )

    print(bigo_result[0])

    values.plot()
    values.to_csv("sizes.csv", index_label="index")

# %%

values["const"] = 1
values["log"] = np.log(values.index)
values["poly"] = values.index**4
values["linear"] = values.index
values["expo"] = values.index
values["expo"] = values["expo"].astype(object)
values["expo"] = 2 ** values["expo"]

plot = (values[0] / values[0].max()).plot()

(values["const"] / values["const"].max()).plot()
(values["log"] / values["log"].max()).plot()
(values["linear"] / values["linear"].max()).plot()
(values["poly"] / values["poly"].max()).plot()
(values["linear"] / values["linear"].max()).plot()
(values["expo"] / values["expo"].max()).plot()
# plot.set_title(f"Complexity Analysis")
# fig = plot.get_figure()
plot.legend()
