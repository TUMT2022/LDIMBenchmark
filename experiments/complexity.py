# %%
from ldimbenchmark.methods import LILA, MNF, DUALMethod
from ldimbenchmark import LDIMBenchmark

methods = [MNF(), LILA(), DUALMethod()]

methods = [
    "ghcr.io/ldimbenchmark/dualmethod:0.2.2",
    "ghcr.io/ldimbenchmark/lila:0.2.2",
    "ghcr.io/ldimbenchmark/mnf:0.2.2",
]

benchmark = LDIMBenchmark(
    {"lila": {"default_flow_sensor": "sum"}}, [], results_dir="./benchmark-results"
)
benchmark.add_local_methods(methods)

results = benchmark.run_complexity_analysis(
    methods=methods,
    style="time",
)


results


# benchmark.run_complexity_analysis(
#     methods=methods,
#     style="junctions",
# )

# %%
