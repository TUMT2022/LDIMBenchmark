# %%

import pandas as pd

stats = pd.read_csv(
    "grid-search/runner_results/lila_0.1.42_battledim-789426bdd3a2afb33b904b293765c5fd_evaluation_fc810cb4fcac3b5edff336f83aa3c72b/stats.csv"
)


# stats[stats["networks"].isna()]["networks"] = "{}"

# # Convert string columns to Dictionary columns
stats["pids_stats"] = stats["pids_stats"].apply(lambda x: eval(x))
stats["blkio_stats"] = stats["blkio_stats"].apply(lambda x: eval(x))
stats["cpu_stats"] = stats["cpu_stats"].apply(lambda x: eval(x))
stats["precpu_stats"] = stats["precpu_stats"].apply(lambda x: eval(x))
stats["memory_stats"] = stats["memory_stats"].apply(lambda x: eval(x))
# stats["networks"] = stats["networks"].apply(lambda x: eval(x))

stats


# %%
flat_stats = pd.json_normalize(stats.to_dict(orient="records"))
flat_stats.columns
flat_stats[
    [
        "memory_stats.usage",
        "memory_stats.max_usage",
        "memory_stats.stats.active_anon",
        "memory_stats.stats.active_file",
        "memory_stats.stats.cache",
        "memory_stats.stats.dirty",
        "memory_stats.stats.hierarchical_memory_limit",
        "memory_stats.stats.hierarchical_memsw_limit",
        "memory_stats.stats.inactive_anon",
        "memory_stats.stats.inactive_file",
        "memory_stats.stats.mapped_file",
        "memory_stats.stats.pgfault",
        "memory_stats.stats.pgmajfault",
        "memory_stats.stats.pgpgin",
        "memory_stats.stats.pgpgout",
        "memory_stats.stats.rss",
        "memory_stats.stats.rss_huge",
        "memory_stats.stats.total_active_anon",
        "memory_stats.stats.total_active_file",
        "memory_stats.stats.total_cache",
        "memory_stats.stats.total_dirty",
        "memory_stats.stats.total_inactive_anon",
        "memory_stats.stats.total_inactive_file",
        "memory_stats.stats.total_mapped_file",
        "memory_stats.stats.total_pgfault",
        "memory_stats.stats.total_pgmajfault",
        "memory_stats.stats.total_pgpgin",
        "memory_stats.stats.total_pgpgout",
        "memory_stats.stats.total_rss",
        "memory_stats.stats.total_rss_huge",
        "memory_stats.stats.total_unevictable",
        "memory_stats.stats.total_writeback",
        "memory_stats.stats.unevictable",
        "memory_stats.stats.writeback",
        "memory_stats.limit",
    ]
]


flat_stats[["memory_stats.usage", "memory_stats.max_usage"]].plot()

# %%

flat_stats[
    [
        "blkio_stats.io_service_bytes_recursive",
        "blkio_stats.io_serviced_recursive",
        "blkio_stats.io_queue_recursive",
        "blkio_stats.io_service_time_recursive",
        "blkio_stats.io_wait_time_recursive",
        "blkio_stats.io_merged_recursive",
        "blkio_stats.io_time_recursive",
        "blkio_stats.sectors_recursive",
    ]
]
