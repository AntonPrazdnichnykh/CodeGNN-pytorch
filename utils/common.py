import subprocess
from os.path import join, dirname
from warnings import filterwarnings
from math import ceil
from omegaconf import DictConfig


SEPARATOR = "|"

# data storage keys
LABEL = "label"
AST = "AST"
SOURCE = "SOURCE"
NODE = "node"
TOKEN = "token"
CHILDREN = "children"

SPLIT_FIELDS = [LABEL, TOKEN]

# vocabulary keys
PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
UNK = "<UNK>"


def get_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path], capture_output=True, encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(f"Counting lines in {file_path} failed with error\n{command_result.stderr}")
    return int(command_result.stdout.split()[0])


def download_dataset(url: str, dataset_dir: str, dataset_name: str):
    download_dir = dirname(dataset_dir)
    download_command_result = subprocess.run(["wget", url, "-P", download_dir], capture_output=True, encoding="utf-8")
    if download_command_result.returncode != 0:
        raise RuntimeError(f"Failed to download dataset. Error:\n{download_command_result.stderr}")
    tar_name = join(download_dir, f"{dataset_name}.tar.gz")
    untar_command_result = subprocess.run(
        ["tar", "-xzvf", tar_name, "-C", download_dir], capture_output=True, encoding="utf-8"
    )
    if untar_command_result.returncode != 0:
        raise RuntimeError(f"Failed to untar dataset. Error:\n{untar_command_result.stderr}")


def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.utilities.distributed", lineno=50)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=216)  # save
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=234)  # load


# def print_config(config: DictConfig, ignore_keys: List[str] = None, n_cols: int = 4):
#     if ignore_keys is None:
#         ignore_keys = []
#     parameters = [f"{k}: {v}" for k, v in config.items() if k not in ignore_keys]
#     table_data = {}
#     items_per_col = int(ceil(len(parameters) / n_cols))
#     for col in range(n_cols):
#         table_data[f"Parameters #{col + 1}"] = parameters[items_per_col * col : items_per_col * (col + 1)]
#     print_table(table_data)
