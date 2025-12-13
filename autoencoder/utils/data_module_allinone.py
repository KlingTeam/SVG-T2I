"""Data Module following the original structure"""

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import numpy as np
import torch
import torchvision

from torchvision import transforms
from PIL import Image
import functools

from ldm.util import instantiate_from_config
from functools import partial
from ldm.data.base import Txt2ImgIterableBaseDataset
from data import DataNoReportException, ItemProcessor, MyDataset, read_general
from utils.imgproc import generate_crop_size_list, to_rgb_if_rgba, var_center_crop


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    """Worker init function - exactly as in original code"""
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


#############################################################################
#                            Data item Processor                            #
#############################################################################

class NonRGBError(DataNoReportException):
    pass

class T2IItemProcessorJson(ItemProcessor):
    def __init__(self, transform):
        self.image_transform = transform
        self.special_format_set = set()

    def process_item(self, data_item, training_mode=False):
        if "path" in data_item:
            url = data_item["path"]
            image = Image.open(read_general(url))
        else:
            raise ValueError(f"Unrecognized item: {data_item}")
            
        if image.mode.upper() != "RGB":
            mode = image.mode.upper()
            if mode not in self.special_format_set:
                self.special_format_set.add(mode)
                print(mode, url)
            if mode == "RGBA":
                image = to_rgb_if_rgba(image)
            elif mode == "P" or mode == "L":
                image = image.convert("RGB")
            else:
                raise NonRGBError()
            
        image = self.image_transform(image)


        return image


class BaseDataModule(pl.LightningDataModule):
    """Common base class for all DataModules."""

    def __init__(self, batch_size, num_workers=None, wrap=False, use_worker_init_fn=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.wrap = wrap
        self.use_worker_init_fn = use_worker_init_fn
        self.dataset_configs = {}
        self.datasets = {}

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = {k: instantiate_from_config(v) for k, v in self.dataset_configs.items()}
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    # === override in subclasses ===
    def create_train_dataset(self):
        raise NotImplementedError

    def collate_fn(self, samples):
        return [x for x in samples]

    def _train_dataloader(self):
        dataset = self.create_train_dataset()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


class DataModuleFromConfigJson(BaseDataModule):
    def __init__(self, batch_size, train_resol=None, json_path=None, **kwargs):
        super().__init__(batch_size, **kwargs)
        self.train_resol = train_resol
        self.json_path = json_path
        self.train_dataloader = self._train_dataloader

    def create_train_dataset(self):
        max_ratio = 4.0
        downsample_ratio = 16
        patch_size = downsample_ratio * 1
        max_num_patches = round((self.train_resol / patch_size) ** 2)
        crop_size_list = generate_crop_size_list(max_num_patches, patch_size, max_ratio)

        print("List of crop sizes:")
        for i in range(0, len(crop_size_list), 6):
            print(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i:i + 6]]))

        image_transform = transforms.Compose([
            transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        dataset = MyDataset(
            self.json_path,
            item_processor=T2IItemProcessorJson(image_transform),
        )
        return dataset



class DataModuleFromConfig(BaseDataModule):
    """Original LDM DataModule (kept for backward compatibility)"""

    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 shuffle_test_loader=False, shuffle_val_dataloader=False, **kwargs):
        super().__init__(batch_size, **kwargs)

        self.shuffle_val_dataloader = shuffle_val_dataloader
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = functools.partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = functools.partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

    def create_train_dataset(self):
        return self.datasets["train"]

    def _train_dataloader(self):
        from ldm.data.base import Txt2ImgIterableBaseDataset
        is_iterable = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if (is_iterable or self.use_worker_init_fn) else None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                        num_workers=self.num_workers, shuffle=not is_iterable, worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        from ldm.data.base import Txt2ImgIterableBaseDataset
        ds = self.datasets['validation']
        init_fn = worker_init_fn if isinstance(ds, Txt2ImgIterableBaseDataset) or self.use_worker_init_fn else None
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                        worker_init_fn=init_fn, shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        from ldm.data.base import Txt2ImgIterableBaseDataset
        ds = self.datasets['test']
        is_iterable = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        init_fn = worker_init_fn if (is_iterable or self.use_worker_init_fn) else None
        shuffle = shuffle and (not is_iterable)
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers,
                        worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        from ldm.data.base import Txt2ImgIterableBaseDataset
        ds = self.datasets['predict']
        init_fn = worker_init_fn if isinstance(ds, Txt2ImgIterableBaseDataset) or self.use_worker_init_fn else None
        return DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn)
