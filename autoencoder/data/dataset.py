from abc import ABC, abstractmethod
import copy
import json
import logging
import os
from pathlib import Path
import random
from time import sleep
import traceback
import warnings

import h5py
import torch.distributed as dist
from torch.utils.data import Dataset
import yaml

logger = logging.getLogger(__name__)


class DataBriefReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class DataNoReportException(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return f"{self.__class__}: {self.message}"


class ItemProcessor(ABC):
    @abstractmethod
    def process_item(self, data_item, training_mode=False):
        raise NotImplementedError


class MyDataset(Dataset):
    def __init__(self, json_path, item_processor):
        logger.info(f"read dataset config from {json_path}")
        logger.info("DATASET CONFIG:")

        self.json_path = json_path
        self.data = self._load_json(json_path)
        self.data_length = len(self.data)

        logger.info(f"total length: {len(self)}")

        self.item_processor = item_processor

    def _load_json(self, path):
        ext = os.path.splitext(path)[-1]
        data = []

        if ext == ".json":
            with open(path, "r") as f:
                data = json.load(f)

        elif ext == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))

        else:
            raise ValueError(f"Unsupported data format: {ext}")

        assert len(data) > 0, "Empty json/jsonl file"
        return data

    def __len__(self):
        return self.data_length

    def get_item_func(self, index):
        data_item = self.data[index]
        return self.item_processor.process_item(data_item, training_mode=True)

    def __getitem__(self, index):
        try:
            # 正常读取逻辑
            data = self.get_item_func(index)
            return data
        except Exception as e:
            logger.warning(f"[Warning] Failed to load item {index}: {e}")
            # fallback to next index
            return self.__getitem__((index + 1) % len(self))
