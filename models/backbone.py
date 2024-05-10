# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # conv1.weight
        # layer1...
        # layer2...
        # layer3...
        # layer4...
        # fc.weight
        # fc.bias
        # 第一个层不需要更新
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                # train_backbone = True时 除了2 3 4 层，其他的层的不需要计算梯度
                parameter.requires_grad_(False)

        if return_interm_layers:
            # 返回三层的特征
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
            # self.num_channels = [128, 256, 512]
        else:
            # 仅返回最后一层的特征
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
            # self.num_channels = [512]
        # IntermediateLayerGetter 构建的backbone，其内部实现在构建时，到layer4之，后面的fc层就跳过了
        # 因此最后是特征，不会是分类头的输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        # xs 包括最后2 3 4 层的输出
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        # xs 可能包含了多层，目标检测任务只使用layer4，分割任务会使用layer1,2,3,4
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 分割任务的mask在不同的特征图尺寸上的插值
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class BackboneBase_T(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        # conv1.weight
        # layer1...
        # layer2...
        # layer3...
        # layer4...
        # fc.weight
        # fc.bias
        # 第一个层不需要更新
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                # train_backbone = True时 除了2 3 4 层，其他的层的不需要计算梯度
                parameter.requires_grad_(False)

        if return_interm_layers:
            # 返回三层的特征
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            # self.num_channels = [128, 256, 512]
            self.num_channels = [512, 1024, 2048]
        else:
            # 仅返回最后一层的特征
            return_layers = {'layer4': "0"}
            self.strides = [32]
            # self.num_channels = [512]
            self.num_channels = [2048]
        # IntermediateLayerGetter 构建的backbone，其内部实现在构建时，到layer4之，后面的fc层就跳过了
        # 因此最后是特征，不会是分类头的输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        # xs 包括最后2 3 4 层的输出
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        # xs 可能包含了多层，目标检测任务只使用layer4，分割任务会使用layer1,2,3,4
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 分割任务的mask在不同的特征图尺寸上的插值
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        # assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class Backbone_T(BackboneBase_T):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        # assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        # 经过backbone
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        # 每个特征层应该是，还是每个image
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    # 创建位置编码
    position_embedding = build_position_encoding(args)
    # 如果backbone有学习率设置，那么backbone也将跟着进行训练
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone_RGB = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    backbone_T = Backbone_T("resnet50", train_backbone, return_interm_layers, args.dilation)
    model_RGB = Joiner(backbone_RGB, position_embedding)
    model_T = Joiner(backbone_T, position_embedding)

    return model_RGB, model_T
