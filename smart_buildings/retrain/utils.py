from typing import Dict

import polars as pl
import yaml


def read_data(path: str) -> pl.DataFrame:
    data = pl.read_csv(path, has_header=False, skip_rows=1)
    header = pl.read_csv(path, n_rows=1, has_header=False, truncate_ragged_lines=True)
    data = data[:, 1:]
    data.columns = header.row(0)
    return data


def get_config(path: str) -> Dict[str, str]:
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config
