"""
Constants used in the ldimbenchmark package.
"""
import os


LDIM_BENCHMARK_CACHE_DIR = ".ldim_benchmark_cache"
CPU_COUNT = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
