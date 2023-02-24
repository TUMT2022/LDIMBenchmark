# Troubleshooting

## My Methods is slower when executed in parallel

You might be using hardware accelarated function in [`numpy`](https://numpy.org/install/#numpy-packages--accelerated-linear-algebra-libraries) or other libraries (commonly BLAS or MKL).
When we run methods in parallel which is inferring with theses libraries (thus hanging).

You should use the following implementation when using these libraries:

e.g. for `numpy`:

```python
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1, user_api="blas"):


```

# My Method hangs when executed in parallel

Because of the dataset sizes used much RAM might be reserved during the execution.
As the process is parallelized the each process might see the entire RAM as free before trying to commit it.
This might lead to a situation where all processes together try to commit more RAM than is available.
This can be solved by limiting the number of parallel threads.

```python
    benchmark = LDIMBenchmark(
        hyperparameters,
        datasets,
        results_dir="./benchmark-results",
    )
    benchmark.add_local_methods(local_methods)
    benchmark.run_benchmark(parallel=True, parallel_max_workers=4)
```
