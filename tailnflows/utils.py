import os
from pathlib import Path
import subprocess
import pickle
from typing import Any
import torch
import multiprocessing as mp
import io


def get_project_root():
    root_path = os.environ.get("TAILNFLOWS_HOME", Path(__file__).parent.parent)
    return root_path

def get_data_path():
    return os.environ.get("TAILNFLOWS_DATA_HOME", f"{get_project_root()}/data")

def get_experiment_output_path():
    return os.environ.get("TAILNFLOWS_EXPERIMENT_DIR", f"{get_project_root()}/experiment_output")

def add_experiment_output_data(path: str, label: str, data: Any, force_write: bool = False) -> None:
    rd_path = f"{get_experiment_output_path()}/{path}.p"

    data_file = Path(rd_path)

    if not data_file.is_file():
        data_file.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({}, open(rd_path, "wb"))
    elif not force_write:
        confirm = input("Experiment data already present, reset? (y/n)")
        if confirm == "y":
            pickle.dump({}, open(rd_path, "wb"))
        else:
            print("no reset, appending data...")

    raw_data = pickle.load(open(rd_path, "rb"))
    if label not in raw_data:
        raw_data[label] = []
    raw_data[label].append(data)
    pickle.dump(raw_data, open(rd_path, "wb"))

def add_raw_data(path: str, label: str, data: Any, force_write: bool = False) -> int:
    rd_path = f"{path}.p"

    data_file = Path(rd_path)

    if not data_file.is_file():
        data_file.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump({}, open(rd_path, "wb"))
    elif not force_write:
        confirm = input("Experiment data already present, reset? (y/n)")
        if confirm == "y":
            pickle.dump({}, open(rd_path, "wb"))
        else:
            print("no reset, appending data...")

    raw_data = pickle.load(open(rd_path, "rb"))
    if label not in raw_data:
        raw_data[label] = []

    ix = len(raw_data[label])
    raw_data[label].append(data)
    pickle.dump(raw_data, open(rd_path, "wb"))
    return ix

class CPU_Unpickler(pickle.Unpickler):
    # thanks https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_experiment_output_data(path: str) -> Any:
    rd_path = f"{get_project_root()}/experiment_output/{path}.p"
    try:
        return pickle.load(open(rd_path, "rb"))
    except RuntimeError:
        return CPU_Unpickler(open(rd_path, "rb")).load()
    
def load_raw_data(path: str) -> Any:
    rd_path = f"{path}.p"
    try:
        return pickle.load(open(rd_path, "rb"))
    except RuntimeError:
        return CPU_Unpickler(open(rd_path, "rb")).load()


def load_torch_data(path: str) -> Any:
    rd_path = f"{get_project_root()}/data/{path}.p"
    if not torch.cuda.is_available():
        data = torch.load(open(rd_path, "rb"), map_location=torch.device("cpu"))
    else:
        data = torch.load(open(rd_path, "rb"), map_location=torch.device("cpu"))

    return data


class RunWrapper:
    def __init__(self, run_experiment):
        self.run_experiment = run_experiment

    def __call__(self, exp_ix_kwargs):
        exp_ix, kwargs = exp_ix_kwargs
        self.run_experiment(experiment_ix=exp_ix + 1, **kwargs)


def parallel_runner(run_experiment, experiments, max_runs=3):
    print(f"{len(experiments)} experiments to run...")
    with mp.Pool(max_runs) as p:
        p.map(RunWrapper(run_experiment), list(enumerate(experiments)), chunksize=1)
