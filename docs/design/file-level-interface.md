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
 | -- dma.json             # Layout of the district metering zones it contains all nodes and pipes in the an area, enabling methods to be specific to each area.
 | -- options.yml # Options for the algorithm (e.g. training and evaluation data timestampy, stage of the algorithm [training, detection_offline, detection_online] and goal (detection, localization), hyperparameters, etc.)
```

> We trust the implementation of the Leakage detection method to use the correct implementation for each stage (e.g. doing online detection if told to instead of offline detection)

The following assumptions are made about the input data:

- the model is the leading datasource (meaning any sensor names in the other files must be present in the model)
  - The name of the csv files is corresponding to the name of the sensor in the inp file.
- the model is a valid EPANET model

Maybe:

- the model might contain existing patterns

The following assumptions are not made:

- Timestamps are not required to be the same for all input files, to make it possible for the methods to do their own resample and interpolation of the data

#### Arguments

```
./args
 | -- options.yml   # Options for running the Method Runner.

```

#### Output:

```
./output
 | -- detected_leaks.csv # The leaks found by the method
 | -- should_have_detected_leaks.csv
 | -- run_info.csv
 | -- debug
 | --  | -- ...      # Any information the method seems suitable as debug information. If the information should be plotted by the evaluation Methods the Timestamps should be the roughly the same as in the supplied dataset.
```
