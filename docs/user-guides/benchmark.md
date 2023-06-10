# Benchmark 

This guide will show you how to prepare your own benchmark.


## Prepare Dataset(s)

## Prepare Benchmark

- Select Methods to Benchmark

Run a Grid Search in order to select the best parameters for the further benchmark.

```python

# Select ranges to test
param_grid = {
    "method_name": {
        "dataset_name": {
            "param_1": np.arange(24, 24 * 8, 24).tolist(),
            "param_2": np.arange(2, 16, 1).tolist(),
            "param_3": np.arange(4, 14, 1).tolist(),
        },
    },
}

benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets=datasets,
    results_dir="./grid-search",
    multi_parameters=True,
)
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/mnf:0.1.20"])

# Choose the training dataset!
benchmark.run_benchmark("training", parallel=True, parallel_max_workers=4)
benchmark.evaluate()
```

## Run the Benchmark

```python

param_grid = {
    "method_name": {
        "dataset_name": {
            # Use best performing parameters from grid-search
            "param_1": np.arange(24, 24 * 8, 24).tolist(),
            "param_2": np.arange(2, 16, 1).tolist(),
            "param_3": np.arange(4, 14, 1).tolist(),
        },
    },
}

benchmark = LDIMBenchmark(
    hyperparameters=param_grid,
    datasets=datasets,
    results_dir="./grid-search",
)
benchmark.add_docker_methods(["ghcr.io/ldimbenchmark/mnf:0.1.20"])

# Choose the evaluation dataset!
benchmark.run_benchmark("evaluation", parallel=True, parallel_max_workers=4)
benchmark.evaluate()
```

Congratulations you successfully ran you first full benchmark locally!

Extra: Run a sensitivity analysis.