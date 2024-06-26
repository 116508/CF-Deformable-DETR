# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher


# 主训练方法
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # 一个epoch内的训练方法
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    # 输出信息间隔
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets, targets_T = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        # 经过网络
        outputs_RGB, outputs_T, Feature_RGB, Feature_T = model(samples)
        # 进行二部图，匈牙利匹配，loss计算
        loss_dict = criterion(outputs_RGB, outputs_T, targets, targets_T, Feature_RGB, Feature_T)
        # loss_dict_RGB = criterion(outputs_RGB, targets)
        # loss_dict_T = criterion(outputs_T, targets_T)
        # 各项loss的权重系数
        # weight_dict = criterion.weight_dict
        # print("RGB_LOSS", loss_dict_RGB)
        # print("T_LOSS", loss_dict_T)
        # 得出loss
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(
            class_error=0.5 * (loss_dict_reduced['class_error_RGB'] + loss_dict_reduced['class_error_T']))
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets, targets_T = prefetcher.next()
        # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator_RGB = CocoEvaluator(base_ds, iou_types)
    coco_evaluator_T = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # for samples, targets_RGB, targets_T in metric_logger.log_every(data_loader, 10, header):
    #     samples_RGB = samples_RGB.to(device)
    #     samples_T = samples_T.to(device)
    #     samples = [samples_RGB, samples_T]
    #     targets_RGB = [{k: v.to(device) for k, v in t.items()} for t in targets_RGB]
    #     targets_T = [{k: v.to(device) for k, v in t.items()} for t in targets_T]

    for samples, targets, targets_T in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets_T = [{k: v.to(device) for k, v in t.items()} for t in targets_T]

        outputs_RGB, outputs_T, Feature_RGB, Feature_T = model(samples)
        # outputs_RGB = {}
        # outputs_T = {}
        # for key, value in outputs.items():
        #     if 'RGB' in key:
        #         outputs_RGB[key.replace('_RGB', '')] = value
        #     else:
        #         outputs_T[key.replace('_T', '')] = value

        loss_dict = criterion(outputs_RGB, outputs_T, targets, targets_T, Feature_RGB, Feature_T)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(
            class_error=0.5 * (loss_dict_reduced['class_error_RGB'] + loss_dict_reduced['class_error_T']))

        orig_target_sizes_RGB = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_RGB = postprocessors['bbox'](outputs_RGB, orig_target_sizes_RGB)
        orig_target_sizes_T = torch.stack([t["orig_size"] for t in targets_T], dim=0)
        results_T = postprocessors['bbox'](outputs_T, orig_target_sizes_T)

        if 'segm' in postprocessors.keys():
            target_sizes_RGB = torch.stack([t["size"] for t in targets], dim=0)
            results_RGB = postprocessors['segm'](results_RGB, outputs_RGB, orig_target_sizes_RGB, target_sizes_RGB)
            target_sizes_T = torch.stack([t["size"] for t in targets_T], dim=0)
            results_T = postprocessors['segm'](results_T, outputs_T, orig_target_sizes_T, target_sizes_T)

        res_RGB = {target['image_id'].item(): output for target, output in zip(targets, results_RGB)}
        res_T = {target['image_id'].item(): output for target, output in zip(targets_T, results_T)}
        if coco_evaluator_RGB and coco_evaluator_T is not None:
            coco_evaluator_RGB.update(res_RGB)
            coco_evaluator_T.update(res_T)

        if panoptic_evaluator is not None:
            res_pano_RGB = postprocessors["panoptic"](outputs_RGB, target_sizes_RGB, orig_target_sizes_RGB)
            res_pano_T = postprocessors["panoptic"](outputs_T, target_sizes_T, orig_target_sizes_T)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano_RGB[i]["image_id"] = image_id
                res_pano_RGB[i]["file_name"] = file_name

            for i, target in enumerate(targets_T):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano_T[i]["image_id"] = image_id
                res_pano_T[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano_RGB)
            panoptic_evaluator.update(res_pano_T)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator_RGB and coco_evaluator_T is not None:
        coco_evaluator_RGB.synchronize_between_processes()
        coco_evaluator_T.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator_RGB and coco_evaluator_T is not None:
        coco_evaluator_RGB.accumulate()
        coco_evaluator_RGB.summarize()
        coco_evaluator_T.accumulate()
        coco_evaluator_T.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if coco_evaluator_RGB and coco_evaluator_T is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox_RGB'] = coco_evaluator_RGB.coco_eval['bbox'].stats.tolist()
            stats['coco_eval_bbox_T'] = coco_evaluator_T.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator_RGB.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator_RGB, coco_evaluator_T

#         losses = sum(0.5 * (loss_dict_RGB[k] + loss_dict_T[k]) * weight_dict[k] for k in loss_dict_RGB.keys() if k in weight_dict)
#
#         # reduce losses over all GPUs for logging purposes
#         # 同步loss
#         loss_dict_reduced_RGB = utils.reduce_dict(loss_dict_RGB)
#         loss_dict_reduced_T = utils.reduce_dict(loss_dict_T)
#         loss_dict_reduced = {}
#         for key in loss_dict_reduced_RGB.keys():
#             loss_dict_reduced[key] = (loss_dict_reduced_RGB[key] + loss_dict_reduced_T[key]) / 2
#         # 没有乘上权重系数
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced_RGB.items()}
#         # 乘上权重系数后的
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#
#         # 总的loss
#         losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
#
#         loss_value = losses_reduced_scaled.item()
#
#         # 判断loss是否正常
#         if not math.isfinite(loss_value):
#             print("Loss is {}, stopping training".format(loss_value))
#             print(losses_reduced_scaled)
#             sys.exit(1)
#
#         # 清空梯度
#         optimizer.zero_grad()
#         # 回传
#         losses.backward()
#         # 防止梯度爆炸
#         if max_norm > 0:
#             grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#         else:
#             grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
#         # 更新
#         optimizer.step()
#
#         # 输出的项目
#         metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#         metric_logger.update(grad_norm=grad_total_norm)
#
#         samples, targets, targets_T = prefetcher.next()
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#
#
# # 主评估方法
# @torch.no_grad()
# def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
#     model.eval()
#     criterion.eval()
#
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
#     header = 'Test:'
#
#     iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
#     coco_evaluator_RGB = CocoEvaluator(base_ds, iou_types)
#     coco_evaluator_T = CocoEvaluator(base_ds, iou_types)
#     # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
#
#     panoptic_evaluator = None
#     if 'panoptic' in postprocessors.keys():
#         panoptic_evaluator = PanopticEvaluator(
#             data_loader.dataset.ann_file,
#             data_loader.dataset.ann_folder,
#             output_dir=os.path.join(output_dir, "panoptic_eval"),
#         )
#
#     for samples, targets, targets_T in metric_logger.log_every(data_loader, 10, header):
#         samples = samples.to(device)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         targets_T = [{k: v.to(device) for k, v in t.items()} for t in targets_T]
#
#         outputs_RGB, outputs_T = model(samples)
#         loss_dict_RGB = criterion(outputs_RGB, targets)
#         loss_dict_T = criterion(outputs_T, targets_T)
#         weight_dict = criterion.weight_dict
#         loss_dict = {}
#         for key in loss_dict_RGB.keys():
#             loss_dict[key] = (loss_dict_RGB[key] + loss_dict_T[key]) / 2
#
#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {k: v * weight_dict[k]
#                                     for k, v in loss_dict_reduced.items() if k in weight_dict}
#         loss_dict_reduced_unscaled = {f'{k}_unscaled': v
#                                       for k, v in loss_dict_reduced.items()}
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])
#
#         orig_target_sizes_RGB = torch.stack([t["orig_size"] for t in targets], dim=0)
#         results_RGB = postprocessors['bbox'](outputs_RGB, orig_target_sizes_RGB)
#         orig_target_sizes_T = torch.stack([t["orig_size"] for t in targets_T], dim=0)
#         results_T = postprocessors['bbox'](outputs_T, orig_target_sizes_T)
#
#         if 'segm' in postprocessors.keys():
#             target_sizes = torch.stack([t["size"] for t in targets], dim=0)
#             results = postprocessors['segm'](results_RGB, outputs_RGB, orig_target_sizes_RGB, target_sizes)
#
#         res_RGB = {target_RGB['image_id'].item(): output_RGB for target_RGB, output_RGB in zip(targets, results_RGB)}
#         res_T = {target_T['image_id'].item(): output_T for target_T, output_T in zip(targets_T, results_T)}
#
#         if coco_evaluator_RGB and coco_evaluator_T is not None:
#             coco_evaluator_RGB.update(res_RGB)
#             coco_evaluator_T.update(res_T)
#
#         if panoptic_evaluator is not None:
#             res_pano = postprocessors["panoptic"](outputs_RGB, target_sizes, orig_target_sizes_RGB)
#             for i, target in enumerate(targets):
#                 image_id = target["image_id"].item()
#                 file_name = f"{image_id:012d}.png"
#                 res_pano[i]["image_id"] = image_id
#                 res_pano[i]["file_name"] = file_name
#
#             panoptic_evaluator.update(res_pano)
#
#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     if coco_evaluator_RGB and coco_evaluator_T is not None:
#         coco_evaluator_RGB.synchronize_between_processes()
#         coco_evaluator_T.synchronize_between_processes()
#     if panoptic_evaluator is not None:
#         panoptic_evaluator.synchronize_between_processes()
#
#     # accumulate predictions from all images
#     if coco_evaluator_RGB and coco_evaluator_T is not None:
#         coco_evaluator_RGB.accumulate()
#         coco_evaluator_RGB.summarize()
#         coco_evaluator_T.accumulate()
#         coco_evaluator_T.summarize()
#     panoptic_res = None
#     if panoptic_evaluator is not None:
#         panoptic_res = panoptic_evaluator.summarize()
#
#     stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#
#     if coco_evaluator_RGB and coco_evaluator_T is not None:
#         if 'bbox' in postprocessors.keys():
#             stats['coco_eval_bbox_RGB'] = coco_evaluator_RGB.coco_eval['bbox'].stats.tolist()
#             stats['coco_eval_bbox_T'] = coco_evaluator_T.coco_eval['bbox'].stats.tolist()
#         if 'segm' in postprocessors.keys():
#             stats['coco_eval_masks'] = coco_evaluator_RGB.coco_eval['segm'].stats.tolist()
#     if panoptic_res is not None:
#         stats['PQ_all'] = panoptic_res["All"]
#         stats['PQ_th'] = panoptic_res["Things"]
#         stats['PQ_st'] = panoptic_res["Stuff"]
#     return stats, coco_evaluator_RGB, coco_evaluator_T
