# Leakage Detection and Isolation Method Benchmark

## Design

Execute Algorithm locally, => Code Interface designed after file interface
and as docker container => File based interface

### File level Interface

The file level interface is the low level interface of the benchmark suite.
It is designed to make it easy to implement the interface in any environment (docker, local, remote).

#### Input:

```
./input
 | -- demands/
 |     | -- <sensorname>.csv
 | -- pressures/
 |     | -- <sensorname>.csv
 | -- flows/
 |     | -- <sensorname>.csv
 | -- levels/
 |     | -- <sensorname>.csv
 | -- model.inp            # The water network model
 | -- dma.json             # Layout of the district metering zones
 | -- options.yml # Options for the algorithm (e.g. training and evaluation data timestampy, stage of the algorithm [training, detection_offline, detection_online] and goal (detection, localization), hyperparameters, etc.)
```

> We trust the implementation of the Leakage detection method to use the correct implementation for each stage (e.g. doing online detection if told to instead of offline detection)

The following assumptions are made about the input data:

- the model is the leading datasource (meaning any sensor names in the other files must be present in the model)
- the model is a valid EPANET model

Maybe:

- the model might contain existing patterns

The following assumptions are not made:

- Timestamps are not required to be the same for all input files, to make it possible for the methods to do their own resample and interpolation of the data

#### Output:

```
./output
 | -- leaks.csv    # The leaks found by the algorithm

```

### Programmming Interface

The programming interface abstracts the low level interface and provides a more convenient interface for the user (in any language).
Currently, the programming interface is implemented only in python.

```

```

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

## Leakage detection methods

Take a dataset input and output the leaks found in the dataset.
Should work across languages and environments.

### Python

- Method Implements the Algorithm
- Method Runner is a wrapper around the method that takes care of the input and output

LocalMethodRunner(
method: Method(),
config: Union[str, dict] # Either provide a path to a config file or a dict with the config
data: str # Path to the dataset
)

> Local Method Runner can be used inside a Docker Container

### Docker

DockerMethodRunner(
image: str,
config: Union[str, dict] # Either provide a path to a config file or a dict with the config
)


