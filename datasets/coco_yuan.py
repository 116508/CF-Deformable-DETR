# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
from PIL import Image
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, img_folder_T,ann_file_T,transforms, return_masks, cache_mode=False, local_rank=0, local_size=1, ):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder_T=img_folder_T
        from pycocotools.coco import COCO
        self.coco_T = COCO(ann_file_T)
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        img_T, target_T=self.getT(idx)
        # plt.figure("rgb")
        # plt.imshow(img)
        # plt.show()
        # plt.figure("t")
        # plt.imshow(img_T)
        # plt.show()
        # img_rgb_temp=img
        # img_t_temp=img_T
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        target_T = {'image_id': image_id, 'annotations': target_T}
        img, target = self.prepare(img, target)
        img_T, target_T = self.prepare(img_T, target_T)

        img = F.to_tensor(img)
        img_T= F.to_tensor(img_T)
        img = torch.cat((img,img_T),dim=0)


        if self._transforms is not None:
            img, target ,target_T= self._transforms(img, target, target_T)

        # from torchvision import transforms
        # channel_mean=torch.tensor([0.485, 0.456, 0.406])
        # channel_std=torch.tensor([0.229, 0.224, 0.225])
        # MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
        # STD = [1 / std for std in channel_std]
        # denormalizer = transforms.Normalize(mean=MEAN, std=STD)
        #
        #
        # img_rgb_temp,img_t_temp=torch.split(img,3,dim=0)
        # img_rgb_temp = denormalizer(img_rgb_temp)
        # img_t_temp = denormalizer(img_t_temp)
        # from torchvision import transforms
        # # img_rgb_temp=img_rgb_temp.transpose(1,2)
        # # img_t_temp = img_t_temp.transpose(1, 2)
        # img_rgb_temp=F.to_pil_image(img_rgb_temp)
        # img_t_temp = F.to_pil_image(img_t_temp)
        # out_bbox=target["boxes"]
        # bboxes_scaled = box_cxcywh_to_xyxy(out_bbox)
        # img_w,img_h=img.shape[2],img.shape[1]
        # bboxes_scaled = bboxes_scaled * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(22, 28))  # [11,2]
        # ax.imshow(img_rgb_temp)
        # for (xmin, ymin, xmax, ymax) in bboxes_scaled:
        #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                                fill=False, color='blue', linewidth=1))
        #     # ax.axis('off')
        #     # ax.set_title(CLASSES[probas[idx].argmax()], fontsize=30)
        #
        # fig.tight_layout()  # 自动调整子图来使其填充整个画布
        # plt.show()
        # out_bbox=target_T["boxes"]
        # bboxes_scaled = box_cxcywh_to_xyxy(out_bbox)
        # bboxes_scaled = bboxes_scaled * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(22, 28))  # [11,2]
        # ax.imshow(img_t_temp)
        # for (xmin, ymin, xmax, ymax) in bboxes_scaled:
        #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                                fill=False, color='blue', linewidth=1))
        #     # ax.axis('off')
        #     # ax.set_title(CLASSES[probas[idx].argmax()], fontsize=30)
        #
        # fig.tight_layout()  # 自动调整子图来使其填充整个画布
        # plt.show()




        return img, target, target_T

    def getT(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco_T
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        if 'visible' in path:
            path = 'visible/' + path
        elif 'lwir' in path:
            path = 'lwir/' + path

        img = Image.open(os.path.join(self.img_folder_T, path)).convert('RGB')
        return img, target
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        # T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])


    raise ValueError(f'unknown {image_set}')

def make_coco_transforms_T(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    # mode = 'instances'
    PATHS = {
        "train": (root / "kaist_train", root / "annotations" / f'visible_train_labels.json'),
        "val": (root / "kaist_test", root / "annotations" / f'visible_test_labels.json'),
    }
    PATHS_T = {
        "train": (root / "kaist_train", root / "annotations" / f'ir_train_labels.json'),
        "val": (root / "kaist_test", root / "annotations" / f'ir_test_labels.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    img_folder_T, ann_file_T = PATHS_T[image_set]
    dataset = CocoDetection(img_folder, ann_file,img_folder_T,ann_file_T, transforms=make_coco_transforms(image_set),return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
