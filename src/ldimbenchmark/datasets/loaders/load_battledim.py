"""
Module containing the loader for the Battledim dataset.
"""

import requests
from os import path
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import wntr
import os
from os import path
import yaml
import shutil
import glob
import logging
from ldimbenchmark.datasets.loaders.load_dataset_base import _LoadDatasetBase
import pandas as pd
from ldimbenchmark.classes import BenchmarkLeakageResult
import json


def download_file(args):
    """
    Download a file from a URL and save it to a given path
    """
    url, out_file_path = args
    if os.path.exists(out_file_path):
        logging.info(f"File {out_file_path} already exists.")
        return
    res = requests.get(url)
    os.makedirs(path.dirname(out_file_path), exist_ok=True)
    logging.info(out_file_path)
    if res.status_code == 200:  # http 200 means success
        with open(out_file_path, "wb") as file_handle:  # wb means Write Binary
            file_handle.write(res.content)
    else:
        logging.error("Failed to download file: " + url)


class BattledimDatasetLoader(_LoadDatasetBase):
    @staticmethod
    def download_dataset(downloadPath=None):
        # Download Battledim Data

        URLs = {
            "https://zenodo.org/record/4017659/files/2018_Fixed_Leakages_Report.txt?download=1": "leak_ground_truth/2018_Fixed_Leakages_Report.txt",
            "https://zenodo.org/record/4017659/files/2018_Leakages.csv?download=1": "leak_ground_truth/2018_Leakages.csv",
            "https://zenodo.org/record/4017659/files/2018_SCADA.xlsx?download=1": "2018/SCADA.xlsx",
            "https://zenodo.org/record/4017659/files/2018_SCADA_Demands.csv?download=1": "2018/Demands.csv",
            "https://zenodo.org/record/4017659/files/2018_SCADA_Flows.csv?download=1": "2018/Flows.csv",
            "https://zenodo.org/record/4017659/files/2018_SCADA_Levels.csv?download=1": "2018/Levels.csv",
            "https://zenodo.org/record/4017659/files/2018_SCADA_Pressures.csv?download=1": "2018/Pressures.csv",
            "https://zenodo.org/record/4017659/files/2019_Leakages.csv?download=1": "leak_ground_truth/2019_Leakages.csv",
            "https://zenodo.org/record/4017659/files/2019_SCADA.xlsx?download=1": "2019/SCADA.xlsx",
            "https://zenodo.org/record/4017659/files/2019_SCADA_Demands.csv?download=1": "2019/Demands.csv",
            "https://zenodo.org/record/4017659/files/2019_SCADA_Flows.csv?download=1": "2019/Flows.csv",
            "https://zenodo.org/record/4017659/files/2019_SCADA_Levels.csv?download=1": "2019/Levels.csv",
            "https://zenodo.org/record/4017659/files/2019_SCADA_Pressures.csv?download=1": "2019/Pressures.csv",
            "https://zenodo.org/record/4017659/files/dataset_configuration.yaml?download=1": "dataset_configuration.yaml",
            "https://zenodo.org/record/4017659/files/L-TOWN.inp?download=1": "L-TOWN.inp",
            "https://zenodo.org/record/4017659/files/L-TOWN_Real.inp?download=1": "L-TOWN_Real.inp",
        }

        args = list(
            zip(URLs.keys(), [path.join(downloadPath, file) for file in URLs.values()])
        )

        with Pool(processes=cpu_count() - 1) as p:
            max_ = len(args)
            with tqdm(total=max_) as pbar:
                for result in p.imap_unordered(download_file, args):
                    pbar.update()

    @staticmethod
    def prepare_dataset(unpreparedDatasetPath=None, preparedDatasetPath=None):
        # Preprocess Battledim Data

        os.makedirs(preparedDatasetPath, exist_ok=True)

        wn = wntr.network.WaterNetworkModel(
            path.join(unpreparedDatasetPath, "L-TOWN.inp")
        )
        wntr.network.write_inpfile(wn, path.join(preparedDatasetPath, "L-TOWN.inp"))

        logging.info("Concatening Files")
        copyfiles = glob.glob(path.join(unpreparedDatasetPath, "2018") + "/*.csv")
        for file in copyfiles:
            shutil.copy(
                file,
                os.path.join(unpreparedDatasetPath, os.path.basename(file).lower()),
            )

        copyfiles = glob.glob(path.join(unpreparedDatasetPath, "2019") + "/*.csv")
        for file in copyfiles:
            with open(file, "r") as file_from:
                with open(
                    os.path.join(unpreparedDatasetPath, os.path.basename(file).lower()),
                    "a",
                ) as file_to:
                    file_to.writelines(file_from.readlines()[1:])

        logging.info("Splitting data by sensor")
        copyfiles = glob.glob(path.join(unpreparedDatasetPath) + "/*.csv")
        for file in copyfiles:
            current_file_base_path = os.path.join(
                preparedDatasetPath, os.path.basename(file).lower()[:-4]
            )
            os.makedirs(current_file_base_path)

            table = pd.read_csv(
                file, delimiter=";", decimal=",", index_col=["Timestamp"]
            )
            for column in table.columns:
                table[column].to_csv(
                    os.path.join(current_file_base_path, f"{column}.csv")
                )

        new_leakages = []
        with open(
            os.path.join(unpreparedDatasetPath, "dataset_configuration.yaml"), "r"
        ) as f:
            content = yaml.safe_load(f.read())

            for leak in content["leakages"]:
                if leak == None:
                    continue
                splits = leak.split(",")
                # TODOlater:  Extract leak area and diameter from extra CSVs
                new_leakages.append(
                    {
                        "leak_pipe_id": splits[0].strip(),
                        "leak_time_start": splits[1].strip(),
                        "leak_time_end": splits[2].strip(),
                        "leak_time_peak": splits[5].strip(),
                        "leak_flow_max": float(splits[3].strip()),
                    }
                )

        dataset_info = """
name: battledim
dataset:
  evaluation:
      start: 2019-01-01 00:00:00
      end: 2019-12-31 23:59:59
  training:
      start: 2018-01-01 00:00:00
      end: 2018-12-31 23:59:59
inp_file: L-TOWN.inp
        """
        # Convert info to yaml dictionary
        dataset_info = yaml.safe_load(dataset_info)

        pd.DataFrame(
            new_leakages,
            columns=list(BenchmarkLeakageResult.__annotations__.keys()),
        ).to_csv(
            os.path.join(preparedDatasetPath, "leaks.csv"),
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )

        # Write info to file
        with open(os.path.join(preparedDatasetPath, f"dataset_info.yaml"), "w") as f:
            yaml.dump(dataset_info, f)

        dmas = {
            "DMA_A": [
                "n54",
                "n105",
                "n114",
                "n163",
                "n188",
                "n288",
                "n296",
                "n332",
                "n342",
                "n410",
                "n415",
                "n429",
                "n458",
                "n469",
                "n495",
                "n506",
                "n516",
                "n519",
                "n549",
                "n613",
                "n636",
                "n644",
                "n679",
                "n722",
                "n726",
                "n740",
                "n752",
                "n769",
            ],
            "DMA_B": ["n215", "n229", "p227", "p235"],
            "DMA_C": [
                "n1",
                "n4",
                "n31",
                "n1",
                "n2",
                "n3",
                "n4",
                "n6",
                "n7",
                "n8",
                "n9",
                "n10",
                "n11",
                "n13",
                "n16",
                "n17",
                "n18",
                "n19",
                "n20",
                "n21",
                "n22",
                "n23",
                "n24",
                "n25",
                "n26",
                "n27",
                "n28",
                "n29",
                "n30",
                "n31",
                "n32",
                "n33",
                "n34",
                "n35",
                "n36",
                "n39",
                "n40",
                "n41",
                "n42",
                "n43",
                "n44",
                "n45",
                "n343",
                "n344",
                "n345",
                "n346",
                "n347",
                "n349",
                "n350",
                "n351",
                "n352",
                "n353",
                "n354",
                "n355",
                "n356",
                "n357",
                "n358",
                "n360",
                "n361",
                "n362",
                "n364",
                "n365",
                "n366",
                "n367",
                "n368",
                "n369",
                "n370",
                "n371",
                "n372",
                "n373",
                "n374",
                "n375",
                "n376",
                "n377",
                "n378",
                "n379",
                "n381",
                "n382",
                "n383",
                "n384",
                "n385",
                "n386",
                "n387",
                "n388",
                "n389",
                "PUMP_1",
            ],
        }
        # Write info to file
        with open(os.path.join(preparedDatasetPath, f"dmas.json"), "w") as f:
            json.dump(dmas, f)
