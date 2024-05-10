# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco

import random
from torch.utils.data import Sampler, BatchSampler


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


class PairedRandomSampler(Sampler[int]):
    """Samples elements randomly from a concatenated dataset of paired samples.

    Arguments:
        data_source (Dataset): dataset to sample from. Assumes that it contains paired samples.
    """

    def __init__(self, data_source, replacement: bool = False, generator=None) -> None:
        self.data_source = data_source
        self.num_samples = len(data_source) // 2
        self.replacement = replacement
        self.generator = generator

    def __iter__(self):
        indices = list(range(self.num_samples))
        random.shuffle(indices)
        return iter([indices[i], indices[i] + self.num_samples] for i in range(self.num_samples))

    def __len__(self):
        return self.num_samples


class PairedSequentialSampler(Sampler):
    """Samples elements sequentially from a concatenated dataset of paired samples.

    Arguments:
        data_source (Dataset): dataset to sample from. Assumes that it contains paired samples.
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices = list(range(len(self.data_source) // 2))
        return iter([indices[i], indices[i] + len(self.data_source) // 2] for i in range(len(self.data_source) // 2))

    def __len__(self):
        return len(self.data_source) // 2


class PairedBatchSampler(BatchSampler):
    """Batches elements sequentially from a concatenated dataset of paired samples.

    Arguments:
        sampler (Sampler): sampler that samples elements sequentially from the paired dataset
        batch_size (int): size of batch
        drop_last (bool): if True, drops the last incomplete batch
    """

    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.num_samples = len(sampler) // 2

    def __iter__(self):
        indices = list(self.sampler)
        return iter([indices[i:i+2] for i in range(0, self.num_samples*2, 2)] for _ in range(self.batch_size))

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size