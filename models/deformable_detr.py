# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
import copy

import numpy as np


# copy clone几份
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone_RGB, backbone_T, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False):
        # num_feature_levels, with_box_refine, two_stage 是DeformableDETR新增的
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # DETR加1，这个地方没有加1，同时后面的处理也有些不同
        self.class_embed_RGB = nn.Linear(hidden_dim, num_classes)
        self.class_embed_T = nn.Linear(hidden_dim, num_classes)
        # 因为bbox的坐标是4，这里的output_dim为4
        # 因此使用多层的MLP可以更好的糅合提取必要的信息
        # 相比于类别的输出头，其只需要使用一个Linear就可以了
        self.bbox_embed_RGB = MLP(hidden_dim, hidden_dim, 4, 3)
        self.bbox_embed_T = MLP(hidden_dim, hidden_dim, 4, 3)
        # multi level
        self.num_feature_levels = num_feature_levels

        # 不使用two_stage的情况
        if not two_stage:
            # 这里处理与DETR不同
            # 这里hidden_dim的数量乘以2，在后面会被split成query_embed和tgt
            # 而detr的tgt在初始时是zeros_like
            # 这里的tgt在网络中是共享的
            self.query_embed_RGB = nn.Embedding(num_queries, hidden_dim*2)
            self.query_embed_T = nn.Embedding(num_queries, hidden_dim * 2)

        if num_feature_levels > 1:
            # backbone 有3层的输出 2 3 4, strides is [8,16,32]
            num_backbone_outs = len(backbone_RGB.strides)
            input_proj_list_RGB = []
            input_proj_list_T = []

            for _ in range(num_backbone_outs):
                # 每一层的channel数量都要投影成 给到transformer一样的
                in_channels = backbone_RGB.num_channels[_]
                # input_proj_list_RGB.append(nn.Sequential(
                #     nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                #     nn.GroupNorm(32, hidden_dim),
                # ))
                input_proj_list_RGB.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                # input_proj_list_T.append(nn.Sequential(
                #     nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                #     nn.GroupNorm(32, hidden_dim),
                # ))
                input_proj_list_T.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))


            for _ in range(num_feature_levels - num_backbone_outs):
                # 这里一共需要num_feature_levels=4个特征层，但是num_backbone_outs=3，
                # 那么会有一个是backbone没法提供的，因此这里会这样构造
                # 这里的in_channels在第一次时最后一个channel的宽度2048
                # 3x3卷积会使特征图小一半
                input_proj_list_RGB.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                # input_proj_list_RGB.append(nn.Sequential(
                #     nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                #     nn.GroupNorm(32, hidden_dim),
                # ))
                input_proj_list_T.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                # input_proj_list_T.append(nn.Sequential(
                #     nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                #     nn.GroupNorm(32, hidden_dim),
                # ))
                in_channels = hidden_dim
            self.input_proj_RGB = nn.ModuleList(input_proj_list_RGB)
            self.input_proj_T = nn.ModuleList(input_proj_list_T)
        else:
            # 用的是其中第一个512的
            self.input_proj_RGB = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_RGB.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
            self.input_proj_T = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone_T.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone_RGB = backbone_RGB
        self.backbone_T = backbone_T
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        # todo
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        # 权重初始化
        self.class_embed_RGB.bias.data = torch.ones(num_classes) * bias_value
        self.class_embed_T.bias.data = torch.ones(num_classes) * bias_value
        # 权重初始化
        nn.init.constant_(self.bbox_embed_RGB.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed_RGB.layers[-1].bias.data, 0)
        nn.init.constant_(self.bbox_embed_T.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed_T.layers[-1].bias.data, 0)
        # 权重初始化
        for proj in self.input_proj_RGB:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        for proj in self.input_proj_T:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers

        if with_box_refine:
            # 这里 class_embed 和 bbox_embed 也都是6个
            # 不使用with_box_refine 这两个也都是六个
            # 不过区别在于使用了with_box_refine之后，这六个是不同的，各自独立的
            # 如果没有使用的话，这六个其实是同一个模块，就跟DETR是相同的
            # clone 6个
            self.class_embed = _get_clones(self.class_embed, num_pred)
            # clone 6个
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            # 参数初始化
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            # bias 初始化成了 0 0 -2 -2
            # todo why -2
            nn.init.constant_(self.bbox_embed_RGB.layers[-1].bias.data[2:], -2.0)
            nn.init.constant_(self.bbox_embed_T.layers[-1].bias.data[2:], -2.0)
            # 这里的使用与DETR不同，DETR的class_embed和bbox_embed都只有一个
            # 这里给每层创建了一个（但其实是相同的class_embed,真实的网络还是只有一份的）
            # 这里其实也可以保持跟DETR相同的设计，但是这里这样做的原因是因为在后面bbox的计算部分
            # 需要不同的decoder层的输出上加上相应的对应层的计算出的便宜量，因此这里会是这样的设计
            self.class_embed_RGB = nn.ModuleList([self.class_embed_RGB for _ in range(num_pred)])
            self.class_embed_T = nn.ModuleList([self.class_embed_T for _ in range(num_pred)])
            self.bbox_embed_RGB = nn.ModuleList([self.bbox_embed_RGB for _ in range(num_pred)])
            self.bbox_embed_T = nn.ModuleList([self.bbox_embed_T for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            # class_embed 输出类别的
            self.transformer.decoder.class_embed = self.class_embed
            # todo 源码中应该是缺少了这行代码，这是我添加的
            self.transformer.decoder.bbox_embed = self.bbox_embed
            for box_embed in self.bbox_embed:
                # todo why -2变回0
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        samples_RGB = copy.deepcopy(samples)
        samples_T = copy.deepcopy(samples)
        samples_RGB.tensors = samples.tensors[:, :3, :, :]
        samples_T.tensors = samples.tensors[:, 3:, :, :]
        # features的item数量为3，detr仅有一个
        # 3就是在backbone里面定义了三层，并不是参数配置的
        features_RGB, pos_RGB = self.backbone_RGB(samples_RGB)
        features_T, pos_T = self.backbone_T(samples_T)
        del samples_RGB, samples_T

        # 经过input_proj之后，这里的feature的channel都是256
        srcs_RGB = []
        masks_RGB = []
        srcs_T = []
        masks_T = []

        # 遍历每一个特征层
        for l, feat in enumerate(features_RGB):
            # 分解出特征和mask
            src, mask = feat.decompose()
            # input_proj 是Conv2d，不同的特征层有不同的卷积
            srcs_RGB.append(self.input_proj_RGB[l](src))
            masks_RGB.append(mask)
            assert mask is not None

        for l, feat in enumerate(features_T):
            # 分解出特征和mask
            src, mask = feat.decompose()
            # input_proj 是Conv2d，不同的特征层有不同的卷积
            srcs_T.append(self.input_proj_T[l](src))
            masks_T.append(mask)
            assert mask is not None

        # 4层的举例，这里会用到最后一个
        # Sequential(
        #   (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        #   (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        # )
        # Sequential(
        #   (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        #   (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        # )
        # Sequential(
        #   (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        #   (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        # )
        # Sequential(
        #   (0): Conv2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #   (1): GroupNorm(32, 256, eps=1e-05, affine=True)
        # )

        # 多出backbone特征层的层
        if self.num_feature_levels > len(srcs_RGB):
            _len_srcs = len(srcs_RGB)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    # 这里的还可以承接backbone的特征
                    src = self.input_proj_RGB[l](features_RGB[-1].tensors)
                else:
                    # 这里的只能承接上一层的了，跟backbone的特征已经中间有隔阂了
                    src = self.input_proj_RGB[l](srcs_RGB[-1])

                # 这个是gt的mask
                m = samples.mask
                # 使用插值创建新的mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                # backbone[1] 是位置编码，因为这个最后的src，和gt的mask，并没有经过位置编码
                # 位置编码也是module，这里就是调用了他们的forward方法
                pos_l = self.backbone_RGB[1](NestedTensor(src, mask)).to(src.dtype)
                srcs_RGB.append(src)
                masks_RGB.append(mask)
                pos_RGB.append(pos_l)

        if self.num_feature_levels > len(srcs_T):
            _len_srcs = len(srcs_T)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    # 这里的还可以承接backbone的特征
                    src = self.input_proj_T[l](features_T[-1].tensors)
                else:
                    # 这里的只能承接上一层的了，跟backbone的特征已经中间有隔阂了
                    src = self.input_proj_T[l](srcs_T[-1])

                # 这个是gt的mask
                m = samples.mask
                # 使用插值创建新的mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]

                # backbone[1] 是位置编码，因为这个最后的src，和gt的mask，并没有经过位置编码
                # 位置编码也是module，这里就是调用了他们的forward方法
                pos_l = self.backbone_T[1](NestedTensor(src, mask)).to(src.dtype)
                srcs_T.append(src)
                masks_T.append(mask)
                pos_T.append(pos_l)

        # 如果使用了two_stage 那么传给tansformer的query_embeds是None
        query_embeds = None
        if not self.two_stage:
            # 也是与DETR相同，用的是query_embed.weight
            query_embeds_RGB = self.query_embed_RGB.weight
            query_embeds_T = self.query_embed_T.weight

        # DETR在这个地方的返回值只有两个，第一个是decoder的输出，第二个是encoder的输出
        # 后两个只有在two stage的时候才有值
        # hs [6,bs,300,256]
        # init_reference [bs,300,2] 经过Linear网络 初始生成的参考点的坐标
        # inter_references [6,bs,300,2] 经过各个decoder layer之后的参考点的坐标？
        # hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = self.transformer(srcs_RGB,
        #                                 srcs_T, masks_RGB, masks_T, pos_RGB, pos_T, query_embeds_RGB, query_embeds_T)
        hs_RGB, hs_T, init_reference_RGB, init_reference_T, inter_references_RGB, inter_references_T, \
        enc_outputs_class_RGB, enc_outputs_class_T, enc_outputs_coord_unact_RGB, enc_outputs_coord_unact_T, Feature_RGB, Feature_T= self.transformer(
            srcs_RGB, srcs_T, masks_RGB, masks_T, pos_RGB, pos_T, query_embeds_RGB, query_embeds_T)

        # outputs_classes_RGB = []
        # outputs_coords_RGB = []
        # outputs_classes_T = []
        # outputs_coords_T = []
        # for lvl in range(hs.shape[0]):
        #     if lvl == 0:
        #         # init_reference 是经过Linear生成的 初始的 参考点的坐标
        #         reference = init_reference
        #     else:
        #         # 网络的输出结果，都是对传入decoder的reference_points进行的偏移量修正，因此需要使用上一个的reference_points
        #         reference = inter_references[lvl - 1]
        #
        #     # 先将参考点位取反函数结果，得到的值是真实的图像上的坐标值，修正也是使用的这个坐标系
        #     # 但最后还是会使用sigmoid再次限制到0-1之间
        #     # sigmoid的反函数 [bs,300,2]
        #     reference = inverse_sigmoid(reference)
        #
        #     # 类别的输出 [bs,300,91]
        #     outputs_class_RGB = self.class_embed_RGB[lvl](hs[lvl][:, :300, ...])
        #     outputs_class_T = self.class_embed_T[lvl](hs[lvl][:, 300:, ...])
        #     # [bs,300,4]
        #     tmp_RGB = self.bbox_embed_RGB[lvl](hs[lvl][:, :300, ...])
        #     tmp_T = self.bbox_embed_T[lvl](hs[lvl][:, 300:, ...])
        #
        #     if reference.shape[-1] == 4:
        #         # == 4 的情况，就是中心坐标以及高宽
        #         tmp_RGB += reference[:, :300, ...]
        #         tmp_T += reference[:, 300:, ...]
        #     else:
        #         # == 2 的情况，就是只有中心坐标
        #         assert reference.shape[-1] == 2
        #         # 与前两个维度相加，前两个维度就是高宽
        #         tmp_RGB[..., :2] += reference[:, :300, ...]
        #         tmp_T[..., :2] += reference[:, 300:, ...]
        #
        #     # bbox的输出 [bs,300,4]
        #     outputs_coord_RGB = tmp_RGB.sigmoid()
        #     outputs_coord_T = tmp_T.sigmoid()
        #
        #     outputs_classes_RGB.append(outputs_class_RGB)
        #     outputs_coords_RGB.append(outputs_coord_RGB)
        #     outputs_classes_T.append(outputs_class_T)
        #     outputs_coords_T.append(outputs_coord_T)
        # # 每层都有一个预测的类别，与DETR的不同在于，DETR这些网络结构只有一个，六层通过同一个网络结构
        # # Deformable DETR这些网络结构有六个，每层对应的使用一个网络结构
        # outputs_class_RGB = torch.stack(outputs_classes_RGB)
        # outputs_class_T = torch.stack(outputs_classes_T)
        # # 每层都有一个预测的框
        # outputs_coord_RGB = torch.stack(outputs_coords_RGB)
        # outputs_coord_T = torch.stack(outputs_coords_T)
        outputs_classes_RGB = []
        outputs_coords_RGB = []
        outputs_classes_T = []
        outputs_coords_T = []
        for lvl in range(hs_RGB.shape[0]):
            if lvl == 0:
                reference_RGB = init_reference_RGB
            else:
                reference_RGB = inter_references_RGB[lvl - 1]
            reference_RGB = inverse_sigmoid(reference_RGB)
            outputs_class_RGB = self.class_embed_RGB[lvl](hs_RGB[lvl])
            tmp_RGB = self.bbox_embed_RGB[lvl](hs_RGB[lvl])
            if reference_RGB.shape[-1] == 4:
                tmp_RGB += reference_RGB
            else:
                assert reference_RGB.shape[-1] == 2
                tmp_RGB[..., :2] += reference_RGB
            outputs_coord_RGB = tmp_RGB.sigmoid()
            outputs_classes_RGB.append(outputs_class_RGB)
            outputs_coords_RGB.append(outputs_coord_RGB)

        for lvl in range(hs_T.shape[0]):
            if lvl == 0:
                reference_T = init_reference_T
            else:
                reference_T = inter_references_T[lvl - 1]
            reference_T = inverse_sigmoid(reference_T)
            outputs_class_T = self.class_embed_T[lvl](hs_T[lvl])
            tmp_T = self.bbox_embed_T[lvl](hs_T[lvl])
            if reference_T.shape[-1] == 4:
                tmp_T += reference_T
            else:
                assert reference_T.shape[-1] == 2
                tmp_T[..., :2] += reference_T
            outputs_coord_T = tmp_T.sigmoid()
            outputs_classes_T.append(outputs_class_T)
            outputs_coords_T.append(outputs_coord_T)

        outputs_class_RGB = torch.stack(outputs_classes_RGB)
        outputs_coord_RGB = torch.stack(outputs_coords_RGB)
        outputs_class_T = torch.stack(outputs_classes_T)
        outputs_coord_T = torch.stack(outputs_coords_T)

        # out = {'pred_logits_RGB': outputs_class_RGB[-1], 'pred_logits_T': outputs_class_T[-1], 'pred_boxes_RGB': outputs_coord_RGB[-1], 'pred_boxes_T': outputs_coord_T[-1]}
        # if self.aux_loss:
        #     out['aux_outputs_RGB'] = self._set_aux_loss(outputs_class_RGB, outputs_coord_RGB)
        #     out['aux_outputs_T'] = self._set_aux_loss(outputs_class_T, outputs_coord_T)
        #
        # if self.two_stage:
        #     enc_outputs_coord = enc_outputs_coord_unact_RGB.sigmoid()
        #     out['enc_outputs'] = {'pred_logits': enc_outputs_class_RGB, 'pred_boxes': enc_outputs_coord}
        # return out
        out_RGB = {'pred_logits': outputs_class_RGB[-1], 'pred_boxes': outputs_coord_RGB[-1]}
        out_T = {'pred_logits': outputs_class_T[-1], 'pred_boxes': outputs_coord_T[-1]}

        # 辅助loss
        if self.aux_loss:
            out_RGB['aux_outputs'] = self._set_aux_loss(outputs_class_RGB, outputs_coord_RGB)
            out_T['aux_outputs'] = self._set_aux_loss(outputs_class_T, outputs_coord_T)

        if self.two_stage:
            # 坐标经过sigmoid，保证在0-1内
            enc_outputs_coord = enc_outputs_coord_unact_RGB.sigmoid()
            out_RGB['enc_outputs'] = {'pred_logits': enc_outputs_class_RGB, 'pred_boxes': enc_outputs_coord}
        return out_RGB, out_T, Feature_RGB, Feature_T


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels_RGB(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs_RGB
        src_logits_RGB = outputs_RGB['pred_logits']

        idx = self._get_src_permutation_idx(indices_RGB)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets_RGB, indices_RGB)])
        target_classes = torch.full(src_logits_RGB.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits_RGB.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits_RGB.shape[0], src_logits_RGB.shape[1], src_logits_RGB.shape[2] + 1],
                                            dtype=src_logits_RGB.dtype, layout=src_logits_RGB.layout, device=src_logits_RGB.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits_RGB, target_classes_onehot, num_boxes_RGB, alpha=self.focal_alpha, gamma=2) * src_logits_RGB.shape[1]
        losses = {'loss_ce_RGB': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error_RGB'] = 100 - accuracy(src_logits_RGB[idx], target_classes_o)[0]
        return losses

    def loss_labels_T(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs_T
        src_logits_T = outputs_T['pred_logits']

        idx = self._get_src_permutation_idx(indices_T)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets_T, indices_T)])
        target_classes = torch.full(src_logits_T.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits_T.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits_T.shape[0], src_logits_T.shape[1], src_logits_T.shape[2] + 1],
                                            dtype=src_logits_T.dtype, layout=src_logits_T.layout, device=src_logits_T.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits_T, target_classes_onehot, num_boxes_T, alpha=self.focal_alpha, gamma=2) * src_logits_T.shape[1]
        losses = {'loss_ce_T': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error_T'] = 100 - accuracy(src_logits_T[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality_RGB(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs_RGB['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets_RGB], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error_RGB': card_err}
        return losses

    @torch.no_grad()
    def loss_cardinality_T(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs_T['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets_T], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error_T': card_err}
        return losses

    def loss_boxes_RGB(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs_RGB
        idx = self._get_src_permutation_idx(indices_RGB)
        src_boxes = outputs_RGB['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets_RGB, indices_RGB)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox_RGB'] = loss_bbox.sum() / num_boxes_RGB

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou_RGB'] = loss_giou.sum() / num_boxes_RGB
        return losses

    def loss_boxes_T(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs_T
        idx = self._get_src_permutation_idx(indices_T)
        src_boxes = outputs_T['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets_T, indices_T)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox_T'] = loss_bbox.sum() / num_boxes_T

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou_T'] = loss_giou.sum() / num_boxes_T
        return losses

    def loss_masks_RGB(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs_RGB

        src_idx = self._get_src_permutation_idx(indices_RGB)
        tgt_idx = self._get_tgt_permutation_idx(indices_RGB)

        src_masks = outputs_RGB["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets_RGB]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask_RGB": sigmoid_focal_loss(src_masks, target_masks, num_boxes_RGB),
            "loss_dice_RGB": dice_loss(src_masks, target_masks, num_boxes_RGB),
        }
        return losses

    def loss_masks_T(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs_T

        src_idx = self._get_src_permutation_idx(indices_T)
        tgt_idx = self._get_tgt_permutation_idx(indices_T)

        src_masks = outputs_T["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets_T]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask_T": sigmoid_focal_loss(src_masks, target_masks, num_boxes_T),
            "loss_dice_T": dice_loss(src_masks, target_masks, num_boxes_T),
        }
        return losses

    def loss_contrast(self, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T):
        Feature_RGB = torch.stack(Feature_RGB)
        Feature_T = torch.stack(Feature_T)
        device = Feature_RGB.device
        samples_cost_RGB = []
        samples_cost_T = []
        for lvl in range(Feature_RGB.shape[0]):
            # bs, n, n_heads, n_levels, n_points, _ = samples[lvl].shape
            bs, n_points, _ = Feature_RGB[lvl].shape
            # sample = samples[lvl].view(bs, n*n_heads*n_levels*n_points, 2)
            # # tgt_bbox_center = tgt_bbox[:, :, :2]
            # # sample_cost = torch.cdist(tgt_bbox_center, sample, p=1)
            # # sample_cost = batch_all_triplet_loss(tgt_bbox_center, sample, margin=0.5, squared='false')
            # sample_cost = MSE_loss(sample, tgt_bbox)
            # samples_cost.append(sample_cost)
            sample = Feature_RGB[lvl][:, :n_points // 2, :]
            sample_inter = Feature_RGB[lvl][:, n_points // 2:, :]

            for i in range(bs):
                nei_sample = sample[i]
                nei_sample_inter = sample_inter[i]

                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                logit_scale = self.logit_scale.exp()

                logits_sample = logit_scale * nei_sample @ nei_sample_inter.T
                logits_sample_inter = logit_scale * nei_sample_inter @ nei_sample.T
                label = torch.eye(n_points // 2, device=device).long()
                label = torch.argmax(label, dim=1)
                # label = torch.ones(n_points // 2, device=device).long()
                # labels = label.repeat(bs,1,1)

                dis = (F.cross_entropy(logits_sample, label) + F.cross_entropy(logits_sample_inter, label)) / 2

                # loss_function = nn.MSELoss()
                # dis = loss_function(sample, sample_inter)
                samples_cost_RGB.append(dis)

        for lvl in range(Feature_T.shape[0]):
            # bs, n, n_heads, n_levels, n_points, _ = samples[lvl].shape
            bs, n_points, _ = Feature_T[lvl].shape
            # sample = samples[lvl].view(bs, n*n_heads*n_levels*n_points, 2)
            # # tgt_bbox_center = tgt_bbox[:, :, :2]
            # # sample_cost = torch.cdist(tgt_bbox_center, sample, p=1)
            # # sample_cost = batch_all_triplet_loss(tgt_bbox_center, sample, margin=0.5, squared='false')
            # sample_cost = MSE_loss(sample, tgt_bbox)
            # samples_cost.append(sample_cost)
            sample = Feature_T[lvl][:, :n_points // 2, :]
            sample_inter = Feature_T[lvl][:, n_points // 2:, :]

            for i in range(bs):
                nei_sample = sample[i]
                nei_sample_inter = sample_inter[i]

                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
                logit_scale = self.logit_scale.exp()

                logits_sample = logit_scale * nei_sample @ nei_sample_inter.T
                logits_sample_inter = logit_scale * nei_sample_inter @ nei_sample.T
                label = torch.eye(n_points // 2, device=device).long()
                label = torch.argmax(label, dim=1)
                # label = torch.ones(n_points // 2, device=device).long()
                # labels = label.repeat(bs,1,1)

                dis = (F.cross_entropy(logits_sample, label) + F.cross_entropy(logits_sample_inter, label)) / 2

                # loss_function = nn.MSELoss()
                # dis = loss_function(sample, sample_inter)
                samples_cost_T.append(dis)

        samples_cost_RGB = torch.stack(samples_cost_RGB)
        samples_cost_T = torch.stack(samples_cost_T)
        samples_cost = torch.cat([samples_cost_RGB, samples_cost_T], dim=0)
        losses_contrast = torch.mean(samples_cost)
        losses = {'loss_contrast': 0.05 * losses_contrast}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T, **kwargs):
        loss_map = {
            'labels_RGB': self.loss_labels_RGB,
            'labels_T': self.loss_labels_T,
            'cardinality_RGB': self.loss_cardinality_RGB,
            'cardinality_T': self.loss_cardinality_T,
            'boxes_RGB': self.loss_boxes_RGB,
            'boxes_T': self.loss_boxes_T,
            'masks_RGB': self.loss_masks_RGB,
            'masks_T': self.loss_masks_T,
            'contrast': self.loss_contrast
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T, **kwargs)

    def forward(self, outputs_RGB, outputs_T, targets_RGB, targets_T, Feature_RGB, Feature_T):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # outputs_RGB = {}
        # outputs_T = {}
        #
        # for key, value in outputs.items():
        #     if 'RGB' in key:
        #         outputs_RGB[key.replace('_RGB', '')] = value
        #     else:
        #         outputs_T[key.replace('_T', '')] = value

        outputs_without_aux_RGB = {k: v for k, v in outputs_RGB.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        outputs_without_aux_T = {k: v for k, v in outputs_T.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices_RGB = self.matcher(outputs_without_aux_RGB, targets_RGB)
        indices_T = self.matcher(outputs_without_aux_T, targets_T)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_RGB = sum(len(t["labels"]) for t in targets_RGB)
        num_boxes_RGB = torch.as_tensor([num_boxes_RGB], dtype=torch.float, device=next(iter(outputs_RGB.values())).device)
        num_boxes_T = sum(len(t["labels"]) for t in targets_T)
        num_boxes_T = torch.as_tensor([num_boxes_T], dtype=torch.float, device=next(iter(outputs_T.values())).device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_RGB)
            torch.distributed.all_reduce(num_boxes_T)
        num_boxes_RGB = torch.clamp(num_boxes_RGB / get_world_size(), min=1).item()
        num_boxes_T = torch.clamp(num_boxes_T / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs_RGB, outputs_T, targets_RGB, targets_T, indices_RGB, indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T, **kwargs))

        if 'aux_outputs' in outputs_RGB or 'aux_outputs' in outputs_T:
            for i, (aux_outputs_RGB, aux_outputs_T) in enumerate(
                    zip(outputs_RGB.get('aux_outputs', []), outputs_T.get('aux_outputs', []))):
                indices_RGB = self.matcher(aux_outputs_RGB, targets_RGB)
                indices_T = self.matcher(aux_outputs_T, targets_T)

                for loss in self.losses:
                    if loss == 'masks_RGB' or loss == 'masks_T':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue

                    kwargs = {}
                    if loss == 'labels_RGB' or loss == 'labels_T':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False

                    l_dict = self.get_loss(loss, aux_outputs_RGB, aux_outputs_T, targets_RGB, targets_T, indices_RGB,
                                           indices_T, num_boxes_RGB, num_boxes_T, Feature_RGB, Feature_T, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs_RGB' in outputs_RGB:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets_RGB)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    # num_classes = 4
    num_classes = 2
    # if args.dataset_file == "coco_panoptic":
    #     num_classes = 250
    device = torch.device(args.device)

    backbone_RGB, backbone_T = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone_RGB,
        backbone_T,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce_RGB': args.cls_loss_coef, 'loss_ce_T': args.cls_loss_coef, 'loss_bbox_RGB': args.bbox_loss_coef, 'loss_bbox_T': args.bbox_loss_coef}
    weight_dict['loss_giou_RGB'] = args.giou_loss_coef
    weight_dict['loss_giou_T'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask_RGB"] = args.mask_loss_coef
        weight_dict["loss_dice_RGB"] = args.dice_loss_coef
        weight_dict["loss_mask_T"] = args.mask_loss_coef
        weight_dict["loss_dice_T"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels_RGB', 'labels_T', 'boxes_RGB', 'boxes_T', 'cardinality_RGB', 'cardinality_T', 'contrast']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors







# class SetCriterion(nn.Module):
#     """ This class computes the loss for DETR.
#     The process happens in two steps:
#         1) we compute hungarian assignment between ground truth boxes and the outputs of the model
#         2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
#     """
#     def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
#         """ Create the criterion.
#         Parameters:
#             num_classes: number of object categories, omitting the special no-object category
#             matcher: module able to compute a matching between targets and proposals
#             weight_dict: dict containing as key the names of the losses and as values their relative weight.
#             losses: list of all the losses to be applied. See get_loss for list of available losses.
#             focal_alpha: alpha in Focal Loss
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.matcher = matcher
#         self.weight_dict = weight_dict
#         self.losses = losses
#         self.focal_alpha = focal_alpha
#
#     def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
#         """Classification loss (NLL)
#         targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
#         """
#         assert 'pred_logits' in outputs
#         # [bs,300,91]
#         src_logits = outputs['pred_logits']
#
#         idx = self._get_src_permutation_idx(indices)
#         target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
#         # 填充的都是91
#         target_classes = torch.full(src_logits.shape[:2], self.num_classes,
#                                     dtype=torch.int64, device=src_logits.device)
#         target_classes[idx] = target_classes_o
#
#         # 这里+1 91变成92 [bs,300,92],onehot这个是原始detr中没有的
#         target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
#                                             dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
#         # target_classes is [bs,300] ->unsqueeze(-1)-> [bs,300,1]
#         # scatter_(dim,index,src) 这里的src都是1
#         # target_classes_onehot 原本是 92个0
#         # target_classes 里面大部分都是91,91就是92个位置中的最后一个，会被填上1
#         # 然后在下一个语句中会将最后一个切掉
#         target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
#
#         # [bs,200,91]
#         target_classes_onehot = target_classes_onehot[:,:,:-1]
#         loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
#         losses = {'loss_ce': loss_ce}
#
#         if log:
#             # TODO this should probably be a separate loss, not hacked in this one here
#             losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
#         return losses
#
#     @torch.no_grad()
#     def loss_cardinality(self, outputs, targets, indices, num_boxes):
#         """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
#         This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
#         """
#         # [bs,300,91]
#         pred_logits = outputs['pred_logits']
#         device = pred_logits.device
#         # image中的gt的数量
#         tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
#         # Count the number of predictions that are NOT "no-object" (which is the last class)
#         card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
#
#         # 差值的均数
#         card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
#         # 这个值应该是越小越好，但是变小也不能保证预测的正确性
#         # 并且对于Deformable DETR，它的91而不是92类别的方式，可能并不适用，这个值没有意义
#         losses = {'cardinality_error': card_err}
#         return losses
#
#     def loss_boxes(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
#            targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
#            The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
#         """
#         assert 'pred_boxes' in outputs
#         idx = self._get_src_permutation_idx(indices)
#         # idx 是个tuple，正好将pred_boxes的第一维是batch的image，第二位是每个预测的box(每个图片100个),将这两个维度，直接取出来了
#         src_boxes = outputs['pred_boxes'][idx]
#         # 取出gt的boxes
#         target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
#
#         loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
#
#         losses = {}
#         losses['loss_bbox'] = loss_bbox.sum() / num_boxes
#
#         # giou loss,iou loss的计算需要左上右下的坐标，因此先获取左上右下的坐标
#         loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
#             box_ops.box_cxcywh_to_xyxy(src_boxes),
#             box_ops.box_cxcywh_to_xyxy(target_boxes)))
#         losses['loss_giou'] = loss_giou.sum() / num_boxes
#         return losses
#
#     def loss_masks(self, outputs, targets, indices, num_boxes):
#         """Compute the losses related to the masks: the focal loss and the dice loss.
#            targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
#         """
#         assert "pred_masks" in outputs
#
#         src_idx = self._get_src_permutation_idx(indices)
#         tgt_idx = self._get_tgt_permutation_idx(indices)
#
#         src_masks = outputs["pred_masks"]
#
#         # TODO use valid to mask invalid areas due to padding in loss
#         target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
#         target_masks = target_masks.to(src_masks)
#
#         src_masks = src_masks[src_idx]
#         # upsample predictions to the target size
#         src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
#                                 mode="bilinear", align_corners=False)
#         src_masks = src_masks[:, 0].flatten(1)
#
#         target_masks = target_masks[tgt_idx].flatten(1)
#
#         losses = {
#             "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
#             "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
#         }
#         return losses
#
#     def _get_src_permutation_idx(self, indices):
#         # permute predictions following indices
#         batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
#         src_idx = torch.cat([src for (src, _) in indices])
#         return batch_idx, src_idx
#
#     def _get_tgt_permutation_idx(self, indices):
#         # permute targets following indices
#         batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
#         tgt_idx = torch.cat([tgt for (_, tgt) in indices])
#         return batch_idx, tgt_idx
#
#     def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
#         # 一共四种loss，masks 只有在分割任务中会被使用
#         # cardinality 这个也并不是一个真的loss的计算
#         loss_map = {
#             'labels': self.loss_labels,
#             'cardinality': self.loss_cardinality,
#             'boxes': self.loss_boxes,
#             'masks': self.loss_masks
#         }
#         assert loss in loss_map, f'do you really want to compute {loss} loss?'
#         return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
#
#     def forward(self, outputs, targets):
#         """ This performs the loss computation.
#         Parameters:
#              outputs: dict of tensors, see the output specification of the model for the format
#                 "pred_logtis": [2,300,91],
#                 "pred_boxes": [2,300,4],
#                 "aux_outputs": 前五层decoder的输出
#              list 内容是bs中的images信息
#              targets: list of dicts, such that len(targets) == batch_size.
#                       The expected keys in each dict depends on the losses applied, see each loss' doc
#                 "boxes"
#                 "labels"
#                 "image_id"
#                 "area"
#                 "iscrowd"
#                 "orig_size"
#                 "size"
#         """
#         outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
#
#         # 两个tensor，第一个是detr中的proposal的下标，第二个是image中gt的下标
#         # Retrieve the matching between the outputs of the last layer and the targets
#         indices = self.matcher(outputs_without_aux, targets)
#
#         # 这个bs中总的boxes的数量
#         # Compute the average number of target boxes accross all nodes, for normalization purposes
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         # 后面device的处理，就是要获取到detr输出的outputs中的tensor的device
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
#
#         # Compute all the requested losses
#         losses = {}
#         for loss in self.losses:
#             kwargs = {}
#             losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))
#
#         # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
#         if 'aux_outputs' in outputs:
#             for i, aux_outputs in enumerate(outputs['aux_outputs']):
#                 indices = self.matcher(aux_outputs, targets)
#                 for loss in self.losses:
#                     if loss == 'masks':
#                         # Intermediate masks losses are too costly to compute, we ignore them.
#                         continue
#                     kwargs = {}
#                     if loss == 'labels':
#                         # Logging is enabled only for the last layer
#                         kwargs['log'] = False
#                     l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
#                     l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
#                     losses.update(l_dict)
#
#         # two-stage的时候会有这个输出
#         if 'enc_outputs' in outputs:
#             # 两项内容，pred_logits,pred_boxes 这两个是encoder的输出经过liner得到的
#             enc_outputs = outputs['enc_outputs']
#             bin_targets = copy.deepcopy(targets)
#             for bt in bin_targets:
#                 # 都当成同一类，为了让分配gt时不用考虑类别
#                 # 即使这个label=0是虚假的，后面也计算了类别的loss，但是也没有什么损失
#                 # 这部分的label 慢慢都会预测成0，也不会有其他的预测
#                 # 不过这个地方使用的class_embed bbox_embed 是detr中的那个，不知这样是否有不妥
#                 bt['labels'] = torch.zeros_like(bt['labels'])
#             indices = self.matcher(enc_outputs, bin_targets)
#             for loss in self.losses:
#                 if loss == 'masks':
#                     # Intermediate masks losses are too costly to compute, we ignore them.
#                     continue
#                 kwargs = {}
#                 if loss == 'labels':
#                     # Logging is enabled only for the last layer
#                     kwargs['log'] = False
#                 # 正常的计算loss
#                 l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
#                 l_dict = {k + f'_enc': v for k, v in l_dict.items()}
#                 losses.update(l_dict)
#
#         return losses
#
#
# class PostProcess(nn.Module):
#     """ This module converts the model's output into the format expected by the coco api"""
#
#     @torch.no_grad()
#     def forward(self, outputs, target_sizes):
#         """ Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
#                           For evaluation, this must be the original image size (before any data augmentation)
#                           For visualization, this should be the image size after data augment, but before padding
#         """
#         # [bs,300,91], [bs,300,4]
#         out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
#
#         assert len(out_logits) == len(target_sizes)
#         assert target_sizes.shape[1] == 2
#
#         # [bs,300,91]
#         prob = out_logits.sigmoid()
#         # [bs,100]                                 [bs,,300*91]
#         # 在300个预测-91个类别中选出概率最大的100个
#         topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
#         # [bs,100]
#         scores = topk_values
#         # 属于哪一个预测 [bs,100]
#         topk_boxes = topk_indexes // out_logits.shape[2]
#         # 预测的类别
#         labels = topk_indexes % out_logits.shape[2]
#         # 中心坐标高宽 -> 左上右下 [bs,300,4]
#         boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
#         # [bs,100,1] repeat -> [bs,100,4] 最后标识属于哪个预测的id 重复4次,
#         # 从boxes中取出对应的框
#         # [bs,100,4]
#         boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
#
#         # 图像的高 宽
#         # and from relative [0, 1] to absolute [0, height] coordinates
#         img_h, img_w = target_sizes.unbind(1)
#
#         # 缩放回图像中的框体大小
#         scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#         boxes = boxes * scale_fct[:, None, :]
#
#         results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
#
#         return results
#
#
# class MLP(nn.Module):
#     """ Very simple multi-layer perceptron (also called FFN)"""
#
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         h = [hidden_dim] * (num_layers - 1)
#         self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
#
#     def forward(self, x):
#         for i, layer in enumerate(self.layers):
#             x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
#         return x
#
#
# def build(args):
#     # 91类就是coco中的stuff类别数量
#     num_classes = 4
#     # num_classes = 5 if args.dataset_file != 'coco' else (num_classes + 1)
#     # num_classes = 20 if args.dataset_file != 'coco' else 91
#     if args.dataset_file == "coco_panoptic":
#         num_classes = 250
#     device = torch.device(args.device)
#
#     backbone_RGB, backbone_T = build_backbone(args)
#
#     transformer = build_deforamble_transformer(args)
#     model = DeformableDETR(
#         backbone_RGB,
#         backbone_T,
#         transformer,
#         num_classes=num_classes,
#         num_queries=args.num_queries,
#         num_feature_levels=args.num_feature_levels,
#         aux_loss=args.aux_loss,
#         with_box_refine=args.with_box_refine,
#         two_stage=args.two_stage,
#     )
#     if args.masks:
#         model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
#     matcher = build_matcher(args)
#     weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
#     weight_dict['loss_giou'] = args.giou_loss_coef
#     if args.masks:
#         weight_dict["loss_mask"] = args.mask_loss_coef
#         weight_dict["loss_dice"] = args.dice_loss_coef
#     # TODO this is a hack
#     if args.aux_loss:
#         aux_weight_dict = {}
#         for i in range(args.dec_layers - 1):
#             aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
#         aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
#         weight_dict.update(aux_weight_dict)
#
#     losses = ['labels', 'boxes', 'cardinality']
#     if args.masks:
#         losses += ["masks"]
#     # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
#     criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
#     criterion.to(device)
#     postprocessors = {'bbox': PostProcess()}
#     if args.masks:
#         postprocessors['segm'] = PostProcessSegm()
#         if args.dataset_file == "coco_panoptic":
#             is_thing_map = {i: i <= 90 for i in range(201)}
#             postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
#
#     return model, criterion, postprocessors
