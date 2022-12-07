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
from ldimbenchmark.datasets.load_dataset_base import _LoadDatasetBase


def download_file(args):
    url, out_file_path = args
    res = requests.get(url)
    os.makedirs(path.dirname(out_file_path), exist_ok=True)
    if res.status_code == 200:  # http 200 means success
        with open(out_file_path, "wb") as file_handle:  # wb means Write Binary
            file_handle.write(res.content)
    else:
        logging.error("Failed to download file: " + url)


class BattledimDatasetLoader(_LoadDatasetBase):
    def downloadBattledimDataset(downloadPath=None):
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

    def prepareBattledimDataset(unpreparedDatasetPath=None, preparedDatasetPath=None):
        # Preprocess Battledim Data

        os.makedirs(preparedDatasetPath, exist_ok=True)

        wn = wntr.network.WaterNetworkModel(
            path.join(unpreparedDatasetPath, "L-TOWN.inp")
        )
        wntr.network.write_inpfile(wn, path.join(preparedDatasetPath, "L-TOWN.inp"))

        folder = path.join(preparedDatasetPath, "2018")
        os.makedirs(folder, exist_ok=True)
        copyfiles = glob.glob(path.join(unpreparedDatasetPath, "2018") + "/*.csv")
        for file in copyfiles:
            shutil.copy(file, folder)

        folder = path.join(preparedDatasetPath, "2019")
        os.makedirs(folder, exist_ok=True)
        copyfiles = glob.glob(path.join(unpreparedDatasetPath, "2019") + "/*.csv")
        for file in copyfiles:
            shutil.copy(file, folder)

        new_leakages = []
        with open(
            os.path.join(unpreparedDatasetPath, "dataset_configuration.yaml"), "r"
        ) as f:
            content = yaml.safe_load(f.read())

            for leak in content["leakages"]:
                if leak == None:
                    continue
                splits = leak.split(",")
                new_leakages.append(
                    {
                        "leak_pipe": splits[0].strip(),
                        "leak_start_time": splits[1].strip(),
                        "leak_end_time": splits[2].strip(),
                        "leak_max_flow": float(splits[3].strip()),
                        "leak_peak_time": splits[5].strip(),
                    }
                )

        dataset_info = """
name: battledim
dataset:
  evaluation:
      start: 2019-01-01 00:00
      end: 2019-12-31 23:55
      overwrites:
        filePath: "2019"
        index_column: Timestamp
        decimal: ','
        delimiter: ;
  training:
      start: 2018-01-01 00:00
      end: 2019-12-31 23:55
      overwrites:
        filePath: "2018"
        index_column: Timestamp 
        decimal: ','
        delimiter: ;
inp_file: L-TOWN.inp
        """
        # Convert info to yaml dictionary
        dataset_info = yaml.safe_load(dataset_info)
        dataset_info["leakages"] = new_leakages

        # Write info to file
        with open(os.path.join(preparedDatasetPath, f"dataset_info.yaml"), "w") as f:
            yaml.dump(dataset_info, f)
