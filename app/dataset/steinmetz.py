from tqdm.notebook import tqdm
import numpy as np
import os, requests


def download_data(data_path: str):
    """
    Download the Steinmetz dataset.
    """
    print("Downloading data.")
    fname = []
    for i in range(3):
        fname.append(f"{data_path}steinmetz_part{i}.npz")

    url = [
        "https://osf.io/agvxh/download",
        "https://osf.io/uv3mw/download",
        "https://osf.io/ehmw2/download",
    ]

    for i in tqdm(range(len(url))):
        if not os.path.isfile(fname[i]):
            try:
                r = requests.get(url[i])
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
            else:
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                else:
                    with open(fname[i], "wb") as fid:
                        fid.write(r.content)


def load_data(data_path: str) -> np.ndarray:
    """
    Load dataset from the given path. If dataset does not exist,
    it is downlaoded.
    """
    # download data if not already present.
    if len(os.listdir(data_path)) == 0:
        download_data(data_path)

    # load dataset
    print("Loading data.")
    all_data = np.array([])

    for i in tqdm(range(len(os.listdir(data_path)))):
        all_data = np.hstack(
            (
                all_data,
                np.load(f"{data_path}steinmetz_part{i}.npz", allow_pickle=True)["dat"],
            )
        )

    # Apply corrections to the following time based variables.
    for i in range(len(all_data)):
        all_data[i]["gocue"] += all_data[i]["stim_onset"]
        all_data[i]["response_time"] += all_data[i]["stim_onset"]
        all_data[i]["feedback_time"] += all_data[i]["stim_onset"]
        
    # squeeze all extra dimensions
    for i in range(len(all_data)):
        for k in all_data[i].keys():
            if type(all_data[i][k]) == np.ndarray:
                all_data[i][k] = all_data[i][k].squeeze()
                
    return all_data
