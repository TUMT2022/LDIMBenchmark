# %%

from pathlib import Path
from ldimbenchmark.datasets import Dataset, DatasetLibrary, DATASETS
from ldimbenchmark import (
    LDIMBenchmark,
    LocalMethodRunner,
    DockerMethodRunner,
    FileBasedMethodRunner,
)
from ldimbenchmark.generator import generateDatasetForTimeSpanDays
import os
import logging
import paramiko
import tarfile

from ldimbenchmark.methods.lila import LILA

# read log level from environment variable
logLevel = os.getenv("LOG_LEVEL", "INFO")
numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(level=numeric_level, handlers=[logging.StreamHandler()])
test_data_folder = "test_data"
test_data_folder_datasets = os.path.join("test_data", "datasets")


# runner = FileBasedMethodRunner(
#     LILA(), os.path.join("test_data", "datasets", "battledim"), "out"
# )
# runner.run()


# Download
battledim_dataset = DatasetLibrary(test_data_folder_datasets).download(
    DATASETS.BATTLEDIM
)


generated_dataset_path = os.path.join(test_data_folder_datasets, "synthetic-days-9")
generateDatasetForTimeSpanDays(90, generated_dataset_path)
dataset = Dataset(generated_dataset_path)

# Copy files to server beforehand


# https://stackoverflow.com/a/19974994/9277073
# class MySFTPClient(paramiko.SFTPClient):
#     def put_dir(self, source, target):
#         """Uploads the contents of the source directory to the target path. The
#         target directory needs to exists. All subdirectories in source are
#         created under target.
#         """
#         for item in os.listdir(source):
#             if os.path.isfile(os.path.join(source, item)):
#                 self.put(os.path.join(source, item), "%s/%s" % (target, item))
#             else:
#                 self.mkdir("%s/%s" % (target, item), ignore_existing=True)
#                 self.put_dir(os.path.join(source, item), "%s/%s" % (target, item))

#     def mkdir(self, path, mode=511, ignore_existing=False):
#         """Augments mkdir by adding an option to not fail if the folder exists"""
#         try:
#             super(MySFTPClient, self).mkdir(path, mode)
#         except IOError:
#             if ignore_existing:
#                 pass
#             else:
#                 raise


# dataset_path = "test_data/datasets/battledim/"
# tarfile_path = os.path.join("test_data/datasets/battledim", "archive.tar")
# with tarfile.open(tarfile_path, mode="w|") as tar:
#     files = Path(os.path.join(os.path.abspath(dataset.path))).rglob("*.*")
#     for file in files:
#         relative_path = os.path.relpath(file, os.path.abspath(dataset.path))
#         print(relative_path)
#         with open(file, "rb") as f:
#             info = tar.gettarinfo(fileobj=f)
#             info.name = relative_path
#             tar.addfile(info, f)


# client = paramiko.client.SSHClient()
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# client.connect(
#     "20.29.32.26",
#     22,
#     "azureuser",
# )  # , password=PASSWORD)
# transport = client.get_transport()
# sftp = MySFTPClient.from_transport(transport)
# # sftp.mkdir("/tmp/test/", ignore_existing=True)
# sftp.put(tarfile_path, "/tmp/archive.tar")
# sftp.put_dir("test_data/datasets/battledim", "/tmp/test")
# sftp.close()

# client.exec_command("mkdir /tmp/untared && tar -xvf /tmp/archive.tar -C /tmp/untared")


## End Copy


# hyperparameters =
# runner = DockerMethodRunner(
#     "ghcr.io/ldimbenchmark/mnf:0.1.20",
#     # "ghcr.io/ldimbenchmark/lila:0.1.20",
#     # "ghcr.io/ldimbenchmark/dualmethod:0.1.20",
#     battledim_dataset[0],
#     hyperparameters=hyperparameters,
#     resultsFolder="./benchmark-results/runner_results",
#     debug=True,
#     # docker_base_url="ssh://azureuser@20.29.32.26",
# )

# (detected_leaks) = runner.run()


# %%
hyperparameters = {
    "lila": {
        "est_length": 24,
        "C_threshold": 10000.0,
        "delta": 2.0,
    },
    "dualmethod": {"est_length": 480.0, "C_threshold": 0.2, "delta": 0.2},
}

benchmark = LDIMBenchmark(
    hyperparameters,
    [dataset, battledim_dataset[0]],
    # derivedDatasets[0],
    # dataset,
    results_dir="./benchmark-results",
    debug=True,
)
benchmark.add_docker_methods(
    [
        "ghcr.io/ldimbenchmark/dualmethod:0.1.20",
        "ghcr.io/ldimbenchmark/lila:0.1.20",
        "ghcr.io/ldimbenchmark/mnf:0.1.20",
    ]
)

benchmark.run_benchmark(parallel=True)

benchmark.evaluate(True)
