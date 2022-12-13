# Leakage Detection and Isolation Method Benchmark

> We recommend lowercase names for datasets and methods (`[a-z0-9_]`). They must be globally unique.
> TODO: Enforce

> All measurements need to be in SI units as described in WNTR.

## Design

Execute Algorithm locally, => Code Interface designed after file interface
and as docker container => File based interface


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
