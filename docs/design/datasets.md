## Datasets

Datasets follow this structure:
We do not split the data into training and evaluation data, but instead provide the start and end of the training and evaluation data in the `dataset_info.yml` file (to make it more configurable).

```
./dataset
 | -- demands/
 |    | -- <sensorname>.csv
 | -- pressures/
 |    | -- <sensorname>.csv
 | -- flows/
 |    | -- <sensorname>.csv
 | -- levels/
 |    | -- <sensorname>.csv
 | -- leaks.csv            # Labeled Leaks
 | -- model.inp            # The water network model
 | -- dma.json             # Layout of the district metering zones
 | -- dataset_info.yml     # Information about the dataset
```

The `dataset_info.yml` file contains the following information:

```yaml
name: <name of the dataset> # root name (important for )
derivation: # Filled in by a DatasetDerivator
  <type>: <key>
description: <optional; description of the dataset>
source: <optional;link to source the dataset>
dataset: # required, start and end of evaluation and training data
  evaluation:
    end: '2016-04-12 05:12:00'
    start: '2016-04-12 01:45:00'
  training:
    end: '2016-04-12 01:45:00'
    start: '2016-04-12 01:15:00'
inp_file: Ragnitz_Calibrated_Zones.inp

# Leakages found in the dataset
leakages:
  - description: S1
    leak_start_time: '2016-04-12T01:37:45'
    leak_peak_time: '2016-04-12T01:37:45'
    leak_end_time: '2016-04-12T01:40:30'
    leak_pipe: LHG3879
    leak_max_flow: 15.03
    leak_out_flow: 15.03
```

### Derived Datasets

For testing the robustness of the leakage detection methods, we derive datasets from the original datasets.
These datasets are subject to additional noise, missing data (sensors, timespans), etc.
