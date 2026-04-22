"""
utils.py — 공통 유틸리티
seed 고정, 로거, config 로더
"""
import os
import random
import logging
import yaml
import numpy as np


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name: str = __name__) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def make_dirs(cfg: dict):
    for key in ["model_dir", "report_dir", "figure_dir"]:
        os.makedirs(cfg["output"][key], exist_ok=True)
    os.makedirs(cfg["data"]["processed_path"], exist_ok=True)
