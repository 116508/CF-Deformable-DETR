# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset, PairedRandomSampler, PairedSequentialSampler
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    # 这里其实就是指的resnet，因为只有一个backbone.0 也没有其他的
    parser.add_argument('--lr_backbone_names', default=["backbone_T.0","backbone_RGB.0"], type=str, nargs='+')
    # backbone的学习率，backbone的学习率要小一点，backbone并不是主要的训练目标
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=80, type=int)
    # 40个epoch后学习率下降
    parser.add_argument('--lr_drop', default=[40, 60], type=int)
    # 参数没有被使用
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    # 梯度裁剪 防止梯度爆炸
    # https://blog.csdn.net/Mikeyboi/article/details/119522689
    # https://blog.csdn.net/zhaohongfei_358/article/details/122820992
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # 使用随机梯度下降
    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    # box 精炼
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    # 两阶段
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    # 最后一层使用dilation
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    # sine或者是learned方式的位置编码
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # todo
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    # 如果 > 1，backbone的特征将返回多层的特征
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    # encoder的层数
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    # decoder的层数
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    # 多头的多头数量
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    # todo 这个有消融实验么 300最佳？
    # detr是100个，这里是300，最后还是会top 100，100个目标query
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    # decoder的cross attention 一个目标点使用4个
    parser.add_argument('--dec_n_points', default=4, type=int)
    # encoder的self attention 一个目标点使用4个
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    # 默认使用辅助Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    # 计算匈牙利匹配中的各个loss的权重系数
    # 分类损失的权重值
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    # L1 Loss的权重值
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    # GIoU Loss的权重值
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    # 计算最终的网络loss的各个权重系数
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    # parser.add_argument('--coco_path', default='/data0/fhl/data/kaist_clean_coco/', type=str)
    parser.add_argument('--coco_path', default='/data0/fhl/data/FLIR_shift/RGB_shift/shift_135/', type=str)
    # parser.add_argument('--coco_path', default='/data0/fhl/data/FLIR_shift/RGB_shift/FLIR_30/', type=str)
    # parser.add_argument('--coco_path', default='/data0/fhl/data/FLIR_shift/RGB_shift/FLIR_50/', type=str)
    # parser.add_argument('--coco_path', default='/data1/fhl/data/LLVIP_detr/', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./exps/v50_t18_deformable_detr',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    # 分布式，多卡训练
    utils.init_distributed_mode(args)
    # git信息
    print("git:\n  {}\n".format(utils.get_sha()))

    # 只有分割任务需要冻结
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device(args.device)

    # fix the seed for reproducibility
    # 固定随机数
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 创建模型，匈牙利匹配-loss，后处理模块
    # postprocessors 只有在进行评估的时候会被使用
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    # 模型参数量

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # model, criterion, postprocessors = build_model(args)
    # checkpoint = torch.load('./r50_deformable_detr-checkpoint.pth')
    # cmodel = checkpoint['model']
    # model_dict = {}
    # for name, parameter in cmodel.items():
    #     if  ("transformer.encoder" in name):
    #         pos=name.find("encoder")
    #         tmp=name[:pos+7]+"_T"+name[pos+7:]
    #
    #         model_dict.update({tmp: parameter})
    #         tmp = name[:pos + 7] + "_RGB" + name[pos + 7:]
    #         model_dict.update({tmp: parameter})
    #     if ("backbone" in name):
    #         pos = name.find("backbone")
    #         tmp = name[:pos + 8] + "_T" + name[pos + 8:]
    #
    #         model_dict.update({tmp: parameter})
    #         tmp = name[:pos + 8] + "_RGB" + name[pos + 8:]
    #         model_dict.update({tmp: parameter})
    #     if ("input_proj" in name):
    #         model_dict.update({name + "_T": parameter})
    #         model_dict.update({name + "_RGB": parameter})
    # model.load_state_dict(model_dict, strict=False)
    # model.to(device)
    # for name, parameter in model.named_parameters():
    #     if ("backbone" in name) or ("transformer.encoder" in name) or ("input_proj" in name) or (
    #             "transformer.level_embed" in name):
    #         parameter.requires_grad = False
    # model_without_ddp = model
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)

    # 创建数据集
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # 数据集采样器
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # batch采样器
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # dataloader
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)


    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # 输出模型参数名称
    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        # 其他部分的学习率
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        # backbone的学习率
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        # 线性投影组件的学习率
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    # 优化器
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    # 学习率更新器
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop)

    # 分布式时的model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # base_ds 是给eval使用的

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    # 权重冻结的模型部分加载
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # 输出目录
    output_dir = Path(args.output_dir)
    # 恢复训练
    if args.resume:
        if args.resume.startswith('https'):
            # 网络文件
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 本地文件
            checkpoint = torch.load(args.resume, map_location='cpu')
        # 缺少的key，不需要的key
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

        if len(missing_keys) > 0:
            # 打印缺少的key
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            # 打印不需要的key
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            # optimizer 恢复
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            # lr_scheduler 恢复
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                # lr更新的轮次
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            # 记录之前的轮次数
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
            )
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # 开始训练
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        # 学习率更新
        lr_scheduler.step()
        if args.output_dir:
            # 保存模型的路径
            # 每一轮的结果都会被保存的
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # 每5个轮次，保存一次模型
            # extra checkpoint before LR drop and every 5 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
            if (epoch + 1) % args.lr_drop[0] == 0 or (epoch + 1) % args.lr_drop[1] == 0 or (epoch + 1) % 10 == 0:
                # 带有轮次id的路径
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                # 保存模型
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # 每个轮次后都会执行评估

        test_stats, coco_evaluator_RGB, coco_evaluator_T = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator_RGB and coco_evaluator_T is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator_RGB.coco_eval:
                    # 保持最新的一份评估结果
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        # 每50个epoch保存一个
                        # 不过Deformable DETR 50个epoch也就够了
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator_RGB.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # 创建输出目录
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
