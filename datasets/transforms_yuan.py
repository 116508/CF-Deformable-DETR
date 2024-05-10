# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate

def crop_my(target, region):
    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return  target

def crop(image, target,target_T,region):
    cropped_image = F.crop(image, *region)
    return cropped_image, crop_my(target,region),crop_my(target_T,region)


def hflip(image, target,target_T):
    flipped_image = F.hflip(image)

    _,h, w = image.shape

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)
    target_T = target_T.copy()
    if "boxes" in target_T:
        boxes = target_T["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target_T["boxes"] = boxes

    if "masks" in target_T:
        target_T['masks'] = target_T['masks'].flip(-1)
    return flipped_image, target,target_T

def resize(image, target,target_T, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image.shape[2],image.shape[1]
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    size = get_size((image.shape[2],image.shape[1]), size, max_size)
    rescaled_image = F.resize(image, size)
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in
                   zip((rescaled_image.shape[2], rescaled_image.shape[1]), (image.shape[2], image.shape[1])))
    ratio_width, ratio_height = ratios
    if target is not None:
        target = target.copy()
        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        h, w = size
        target["size"] = torch.tensor([h, w])

        if "masks" in target:
            target['masks'] = interpolate(
                target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
    if target_T is not None:
        target_T = target_T.copy()
        if "boxes" in target_T:
            boxes = target_T["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target_T["boxes"] = scaled_boxes

        if "area" in target_T:
            area = target_T["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target_T["area"] = scaled_area

        h, w = size
        target_T["size"] = torch.tensor([h, w])

        if "masks" in target_T:
            target_T['masks'] = interpolate(
                target_T['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5
    return rescaled_image, target,target_T


def pad(image, target,target_T, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None,None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    target_T = target_T.copy()
    # should we do something wrt the original size?
    target_T["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target_T:
        target_T['masks'] = torch.nn.functional.pad(target_T['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target,target_T


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target,target_T):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target,target_T, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target:dict,target_T: dict):
        w = random.randint(self.min_size, min(img.shape[2], self.max_size))
        h = random.randint(self.min_size, min(img.shape[1], self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target,target_T, region)


# class CenterCrop(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, img, target,target_T):
#         image_width, image_height = img.size
#         crop_height, crop_width = self.size
#         crop_top = int(round((image_height - crop_height) / 2.))
#         crop_left = int(round((image_width - crop_width) / 2.))
#         return crop(img, target,target_T, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target,target_T):
        if random.random() < self.p:
            return hflip(img, target,target_T)
        return img, target,target_T


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target,target_T=None):
        size = random.choice(self.sizes)
        return resize(img, target,target_T, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target,target_T):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target,target_T, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target,target_T):
        if random.random() < self.p:
            return self.transforms1(img, target,target_T)
        return self.transforms2(img, target,target_T)


class ToTensor(object):
    def __call__(self, img, target,target_T):
        return F.to_tensor(img), target,target_T


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target,target_T):
        return self.eraser(img), target,target_T


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None,target_T=None):
        image,image_T=torch.split(image,3,dim=0)
        image = F.normalize(image, mean=self.mean, std=self.std)
        image_T = F.normalize(image_T, mean=self.mean, std=self.std)
        image=torch.cat((image,image_T),dim=0)
        if target is not None:
            target = target.copy()
            h, w = image.shape[-2:]
            if "boxes" in target:
                boxes = target["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target["boxes"] = boxes
        if target_T is not None:
            target_T = target_T.copy()
            if "boxes" in target_T:
                boxes = target_T["boxes"]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
                target_T["boxes"] = boxes
        return image, target,target_T


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target,target_T):
        for t in self.transforms:
            image, target,target_T = t(image, target,target_T)
        return image, target,target_T

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
