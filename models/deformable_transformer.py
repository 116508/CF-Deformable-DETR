# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
# 多尺度可变性Attention模块
from models.ops.modules import MSDeformAttn, MSDeformAttn_cross, MSDeformAttn_cross_interactive, MSDeformAttn_cross_base


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4,
                 # 这两个4应该就是这个论文主要的Deformable的内容
                 dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        # 256
        self.d_model = d_model
        # 8头注意力
        self.nhead = nhead
        # False
        self.two_stage = two_stage
        # 300
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer_RGB = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        encoder_layer_T = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder_RGB = DeformableTransformerEncoder(encoder_layer_RGB, num_encoder_layers)
        self.encoder_T = DeformableTransformerEncoder(encoder_layer_T, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        # decoder_layer_fusion = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
        #                                                   dropout, activation,
        #                                                   num_feature_levels, nhead, dec_n_points)

        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        # deformable detr 多的内容 论文中有提及
        # scale-level position embedding  [4, 256]
        # 因为deformable detr用到了多尺度特征  经过backbone会生成4个不同尺度的特征图  但是如果还是使用原先的sine position embedding
        # detr是针对h和w进行编码的 不同位置的特征点会对应不同的编码值 但是deformable detr不同的特征图的不同位置就有可能会产生相同的位置编码，就无法区分了
        # 为了解决这个问题，这里引入level_embed这个遍历  不同层的特征图会有不同的level_embed 再让原先的每层位置编码+每层的level_embed
        # 这样就很好的区分不同层的位置编码了  而且这个level_embed是可学习的
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # two_stage 使用的内容
        if two_stage:
            # 对Encoder输出memory进行处理：全连接层 + 层归一化
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            # 对top-k proposal box进行处理得到最终的query和query pos
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points_RGB = nn.Linear(d_model, 2)
            self.reference_points_T = nn.Linear(d_model, 2)
        # 参数重制
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points_RGB.weight.data, gain=1.0)
            constant_(self.reference_points_RGB.bias.data, 0.)
            xavier_uniform_(self.reference_points_T.weight.data, gain=1.0)
            constant_(self.reference_points_T.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        # 获取proposal对应的位置嵌入 [bs,300,4]
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        # 0->128
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        # 类似于Transformer的位置嵌入公式
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # sigmoid然后缩放
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    # two stage的情况时会被调用
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """得到第一阶段预测的所有proposal box output_proposals和处理后的Encoder输出output_memory
        memory: Encoder输出特征  [bs, H/8 * W/8 + ... + H/64 * W/64, 256]
        memory_padding_mask: Encoder输出特征对应的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        spatial_shapes: [4, 2] backbone输出的4个特征图的shape
        """
        # bs, all hw, channel
        N_, S_, C_ = memory.shape
        base_scale = 4.0

        proposals = []
        # 定位在memory_padding_mask中的位置，memory_padding_mask这里是所有特征层的内容在一起了
        # 帮助找到mask中每个特征图的初始index
        _cur = 0
        # 某个特征层的 高 宽
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 获取mask, memory_padding_mask is[bs,all hw] [bs,H_,W_,1]
            # 1、生成所有proposal box的中心点坐标xy
            # 展平后的mask [bs, 76, 112, 1]
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            # 有效的高度
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            # 有效的宽度
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            # 生成网格点，这里没有0.5的做法 get_reference_points这个方法中是从0.5开始，这里是从0开始
            # grid_y = [76, 112]   76行112列  第一行全是0  第二行全是1 ... 第76行全是75
            # grid_x = [76, 112]   76行112列  76行全是 0 1 2 ... 111
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            # 生成网格二维点
            # grid = [76, 112, 2(xy)]   这个特征图上的所有坐标点x,y
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            # scale [bs,1,1,2]
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            # 获取相对值 [bs,h,w,2]
            # [76, 112, 2(xy)] -> [1, 76, 112, 2] + 0.5 得到所有网格中心点坐标  这里和one-stage的get_reference_points函数原理是一样的
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # [bs,h,w,2] 宽高的相对值，不同的特征层 wh不同,层级越高，wh相对也会更大
            # 2、生成所有proposal box的宽高wh  第i层特征默认wh = 0.05 * (2**i)
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # [bs,h,w,4] -> [bs,hw,4]
            # 3、concat xy+wh -> proposal xywh [bs, 76x112, 4(xywh)]
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)

            proposals.append(proposal)
            _cur += (H_ * W_)

        # [bs,all hw,4]
        # concat 4 feature map proposals [bs, H/8 x W/8 + ... + H/64 x W/64] = [bs, 11312, 4]
        output_proposals = torch.cat(proposals, 1)
        # [bs,all hw,1] proposals 不需要太靠近边界
        # 筛选一下 xywh 都要处于（0.01，0.99）之间
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # sigmoid的反函数
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # 对于mask部分，填充inf
        # mask的地方是无效的 直接用inf代替
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # 靠近边界的部分，填充inf
        # 再按条件筛选一下 不符合的用用inf代替
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        # 并不修改memory
        output_memory = memory
        # 对于memory的输出进行同样的填充
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        # [bs,all hw,256]
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

        # 对encoder输出进行处理：全连接层 + LayerNorm
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # output_memory 是memory 经过填充inf，经过一层全连接后的结果
        # output_proposals 是制作的proposals，非法的位置填充了inf
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        """
        图像的大小在一个batch内统一成了最大的高宽，但是具体的一个张图会占据其中左上的某个区域，其他的区域在Mask中是True
        这里就要求出真实的图像大小占据的比率
        mask [bs,h,w]
        有效高宽占总的高宽的比率
        """
        # 特征图的高 宽
        _, H, W = mask.shape

        # tensor的size就是bs的大小 [:,:,0] 就是取的第一列，那么就是高度的意思，~mask，有mask的位置是True，~mask取反
        # 有效的高度
        valid_H = torch.sum(~mask[:, :, 0], 1)
        # 有效的宽度
        valid_W = torch.sum(~mask[:, 0, :], 1)
        # 占总长度的比例
        valid_ratio_h = valid_H.float() / H
        # 占总宽度的比例
        valid_ratio_w = valid_W.float() / W

        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    # 这里的内容比原始的DETR多了很多
    def forward(self, srcs_RGB, srcs_T, masks_RGB, masks_T, pos_embeds_RGB, pos_embeds_T, query_embeds_RGB=None, query_embeds_T=None):
        """
        srcs backbone的特征
        masks seg使用的
        pos_embeds 位置编码
        query_embed decoder使用

        经过backbone输出4个不同尺度的特征图srcs，以及这4个特征图对应的masks和位置编码
        srcs: list4  0=[bs,256,H/8,W/8] 1=[bs,256,H/16,W/16] 2=[bs,256,H/32,W/32] 3=[bs,256,H/64,W/64]
        masks: list4  0=[bs,H/8,W/8] 1=[bs,H/16,W/16] 2=[bs,H/32,W/32] 3=[bs,H/64,W/64]
        pos_embeds: list4  0=[bs,256,H/8,W/8] 1=[bs,256,H/16,W/16] 2=[bs,256,H/32,W/32] 3=[bs,256,H/64,W/64]
        query_embed: query embedding 参数 [300, 512]
        """
        # 有two_stage 或者不是two_stage但是要有query_embed
        assert self.two_stage or query_embeds_RGB is not None

        # prepare input for encoder
        # 为encoder的输入作准备：将多尺度特征图、各特征图对应的mask、位置编码、各特征图的高宽、各特征图flatten后的起始索引等展平
        src_flatten_RGB = []
        mask_flatten_RGB = []
        lvl_pos_embed_flatten_RGB = []
        src_flatten_T = []
        mask_flatten_T = []
        lvl_pos_embed_flatten_T = []
        # 特征图的高宽
        spatial_shapes_RGB = []
        spatial_shapes_T = []
        # src内的特征图是从大尺寸到小尺寸
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs_RGB, masks_RGB, pos_embeds_RGB)):
            bs, c, h, w = src.shape
            # 特征图的高宽
            spatial_shape = (h, w)

            spatial_shapes_RGB.append(spatial_shape)
            # [bs,hw,256]
            # 这块维度的顺序与detr不同 detr 是 hw，bs，dim，这里是 bs,hw, dim
            src = src.flatten(2).transpose(1, 2)
            # [bs,hw]
            mask = mask.flatten(1)
            # [bs,hw,256]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # level embed 会被加入到位置编码中,level是在论文中提到过的
            # 这个还多个level pos embed [bs,hw,256]
            # scale-level position embedding  [bs,hxw,c] + [1,1,c] -> [bs,hxw,c]
            # 每一层所有位置加上相同的level_embed 且 不同层的level_embed不同
            # 所以这里pos_embed + level_embed，这样即使不同层特征有相同的w和h，那么也会产生不同的lvl_pos_embed  这样就可以区分了
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten_RGB.append(lvl_pos_embed)
            src_flatten_RGB.append(src)
            mask_flatten_RGB.append(mask)

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs_T, masks_T, pos_embeds_T)):
            bs, c, h, w = src.shape
            # 特征图的高宽
            spatial_shape = (h, w)

            spatial_shapes_T.append(spatial_shape)
            # [bs,hw,256]
            # 这块维度的顺序与detr不同 detr 是 hw，bs，dim，这里是 bs,hw, dim
            src = src.flatten(2).transpose(1, 2)
            # [bs,hw]
            mask = mask.flatten(1)
            # [bs,hw,256]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # level embed 会被加入到位置编码中,level是在论文中提到过的
            # 这个还多个level pos embed [bs,hw,256]
            # scale-level position embedding  [bs,hxw,c] + [1,1,c] -> [bs,hxw,c]
            # 每一层所有位置加上相同的level_embed 且 不同层的level_embed不同
            # 所以这里pos_embed + level_embed，这样即使不同层特征有相同的w和h，那么也会产生不同的lvl_pos_embed  这样就可以区分了
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten_T.append(lvl_pos_embed)
            src_flatten_T.append(src)
            mask_flatten_T.append(mask)

        # [bs,all hw,256]
        # 所有特征层的拼在一起
        # 他们就是在维度1上长度不同，尺寸越大的特征层，1上的数量越多
        # list4[bs, H/8 * W/8, 256] [bs, H/16 * W/16, 256] [bs, H/32 * W/32, 256] [bs, H/64 * W/64, 256] -> [bs, K, 256]
        # K =  H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64
        src_flatten_RGB = torch.cat(src_flatten_RGB, 1)
        src_flatten_T = torch.cat(src_flatten_T, 1)
        # [bs,all hw]
        # list4[bs, H/8 * W/8] [bs, H/16 * W/16] [bs, H/32 * W/32] [bs, H/64 * W/64] -> [bs, K]
        mask_flatten_RGB = torch.cat(mask_flatten_RGB, 1)
        mask_flatten_T = torch.cat(mask_flatten_T, 1)
        # [bs, all hw,256]
        # list4[bs, H/8 * W/8, 256] [bs, H/16 * W/16, 256] [bs, H/32 * W/32, 256] [bs, H/64 * W/64, 256] -> [bs, K, 256]
        lvl_pos_embed_flatten_RGB = torch.cat(lvl_pos_embed_flatten_RGB, 1)
        lvl_pos_embed_flatten_T = torch.cat(lvl_pos_embed_flatten_T, 1)
        # [特征层的数量，2] 存储的是高宽
        # [4, h+w]  4个特征图的高和宽
        spatial_shapes_RGB = torch.as_tensor(spatial_shapes_RGB, dtype=torch.long, device=src_flatten_RGB.device)
        spatial_shapes_T = torch.as_tensor(spatial_shapes_T, dtype=torch.long, device=src_flatten_T.device)
        # 各个src层 起始的位置, 第一个spatial_shapes.new_zeros((1,))是在起始位置填的0
        # 不同尺度特征图对应被flatten的那个维度的起始索引  Tensor[4]  如[0,15100,18900,19850]
        level_start_index_RGB = torch.cat((spatial_shapes_RGB.new_zeros((1, )), spatial_shapes_RGB.prod(1).cumsum(0)[:-1]))
        level_start_index_T = torch.cat((spatial_shapes_T.new_zeros((1,)), spatial_shapes_T.prod(1).cumsum(0)[:-1]))
        # 有效高宽占总的batch高宽的比率 [bs,4,2]
        # 各尺度特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        valid_ratios_RGB = torch.stack([self.get_valid_ratio(m) for m in masks_RGB], 1)
        valid_ratios_T = torch.stack([self.get_valid_ratio(m) for m in masks_T], 1)

        # src_flatten 类似detr中的src，mask_flatten应该是对应了src_key_padding_mask，lvl_pos_embed_flatten对应了pos
        # 其他的三个完全是这里多出来的参数
        # encoder
        # [bs,all hw,256]
        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        memory_RGB = self.encoder_RGB(src_flatten_RGB, spatial_shapes_RGB, level_start_index_RGB, valid_ratios_RGB, lvl_pos_embed_flatten_RGB, mask_flatten_RGB)
        memory_T = self.encoder_T(src_flatten_T, spatial_shapes_T, level_start_index_T, valid_ratios_T,lvl_pos_embed_flatten_T, mask_flatten_T)

        # prepare input for decoder
        # 为decoder的输入作准备：得到参考点、query embedding(tgt)和query pos(query_embed)
        # one-stage和two-stage的生成方式不同
        # two-stage: 参考点=Encoder预测的top-k（300个）得分最高的proposal boxes,然后对参考点进行位置嵌入生成query和query pos
        # one-stage: query和query pos就是预设的query_embed,然后将query_embed经过全连接层输出2d参考点（归一化的中心坐标）
        bs, _, c = memory_RGB.shape
        if self.two_stage:
            # output_memory 是memory 经过填充inf，经过一层全连接后的结果 [bs,all hw,256]
            # output_proposals 是制作的proposals，非法的位置填充了inf [bs,all hw,4]
            # 对memory进行处理得到output_memory: [bs, H/8 * W/8 + ... + H/64 * W/64, 256]
            # 并生成初步output_proposals: [bs, H/8 * W/8 + ... + H/64 * W/64, 4]  其实就是特征图上的一个个的点坐标
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory_RGB, mask_flatten_RGB, spatial_shapes_RGB)

            # hack implementation for two-stage Deformable DETR
            # encoder的输出直接使用最后的分类头得到输出结果
            # 作为一个返回值
            # [bs,all hw,91]
            # hack implementation for two-stage Deformable DETR
            # 多分类：[bs, H/8 * W/8 + ... + H/64 * W/64, 256] -> [bs, H/8 * W/8 + ... + H/64 * W/64, 91]
            # 其实个人觉得这里直接进行一个二分类足够了
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            # encoder的输出直接使用最后的bbox头得到输出结果
            # 作为一个返回值
            # 原始代码这里decoder.bbox_embed 并未赋值 [2,all hw,4]
            # 回归：预测偏移量 + 参考点坐标   [bs, H/8 * W/8 + ... + H/64 * W/64, 4]
            # two-stage 必须和 iterative bounding box refinement一起使用 不然bbox_embed=None 报错
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            # 得到参考点reference_points/先验框
            topk = self.two_stage_num_proposals
            # 前300个proposals
            # 直接用第一个类别的预测结果来算top-k，代表二分类
            # 如果不使用iterative bounding box refinement那么所有class_embed共享参数 导致第二阶段对解码输出进行分类时都会偏向于第一个类别
            # topk_proposals: [bs, 300]  top300 index
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            # 取出对应的300个proposals的预测值 [bs,300,4]
            # topk_coords_unact: top300个分类得分最高的index对应的预测bbox [bs, 300, 4]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            # 脱离gpu
            # 以先验框的形式存在  取消梯度
            topk_coords_unact = topk_coords_unact.detach()
            # 相对值，都压制在0-1内
            # 得到归一化参考点坐标  最终会送到decoder中作为初始的参考点
            reference_points = topk_coords_unact.sigmoid()

            # 生成Docder的query和query pos
            # 先对top-k proposal box进行位置编码，编码方式是给xywh每个都赋予128维 其中每个128维使用sine编码  最后用全连接层和LN处理
            # 最终得到pos_trans_out: [bs, 300, 512] 前256为query pos(x、y信息)  后256为query(w、h信息)
            init_reference_out = reference_points
            # [bs,300,512] 先获取pos_embed，然后经过网络，然后经过layer norm
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            # 分解出query_embed和tgt
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:  # 默认执行
            # query_embed [300,512] -> [300,256] tgt [300,256]
            # 随机初始化 query_embed = nn.Embedding(num_queries, hidden_dim*2)
            # [300, 512] -> [300, 256] + [300, 256]
            query_embed_RGB, tgt_RGB = torch.split(query_embeds_RGB, c, dim=1)
            query_embed_T, tgt_T = torch.split(query_embeds_T, c, dim=1)
            # 扩出第0维度，bs
            # 初始化query pos [300, 256] -> [bs, 300, 256]
            query_embed_RGB = query_embed_RGB.unsqueeze(0).expand(bs, -1, -1)
            query_embed_T = query_embed_T.unsqueeze(0).expand(bs, -1, -1)
            # 扩充第0维度成为bs
            # 初始化query embedding [300, 256] -> [bs, 300, 256]
            tgt_RGB = tgt_RGB.unsqueeze(0).expand(bs, -1, -1)
            tgt_T = tgt_T.unsqueeze(0).expand(bs, -1, -1)
            # [bs,300,2]  reference_points为全连接层 512-2，生成出参考点位坐标
            # 这些坐标经过了sigmoid处理，在最后得到修正的点的坐标时，还会使用sigmoid的反函数
            # 由query pos接一个全连接层 再归一化后的参考点中心坐标 [bs, 300, 256] -> [bs, 300, 2]
            reference_points_RGB = self.reference_points_RGB(query_embed_RGB).sigmoid()
            reference_points_T = self.reference_points_T(query_embed_T).sigmoid()
            # 经过网络初始生成出的参考点坐标
            # 初始化的归一化参考点坐标 [bs, 300, 2]
            init_reference_out_RGB = reference_points_RGB
            init_reference_out_T = reference_points_T
            # init_reference_out = torch.cat([init_reference_out_RGB, init_reference_out_T], dim=1)

        # decoder
        # hs [6,bs,300,256]
        # inter_references [6,bs,300,2]
        # decoder
        # tgt: 初始化query embedding [bs, 300, 256]
        # reference_points: 由query pos接一个全连接层 再归一化后的参考点中心坐标 [bs, 300, 2]  two-stage=[bs, 300, 4]
        # query_embed: query pos[bs, 300, 256]
        # memory: Encoder输出结果 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        # spatial_shapes: [4, 2] 4个特征层的shape
        # level_start_index: [4, ] 4个特征层flatten后的开始index
        # valid_ratios: [bs, 4, 2]
        # mask_flatten: 4个特征层flatten后的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        # hs: 6层decoder输出 [n_decoder, bs, num_query, d_model] = [6, bs, 300, 256]
        # inter_references: 6层decoder学习到的参考点归一化中心坐标  [6, bs, 300, 2]
        #                   one-stage=[n_decoder, bs, num_query, 2]  two-stage=[n_decoder, bs, num_query, 4]
        # hs, inter_references = self.decoder(tgt_RGB, tgt_T, reference_points_RGB, reference_points_T, memory_RGB, memory_T,
        #                                     spatial_shapes_RGB, spatial_shapes_T, level_start_index_RGB, level_start_index_T,
        #                                     valid_ratios_RGB, valid_ratios_T, query_embed_RGB, query_embed_T, mask_flatten_RGB, mask_flatten_T)

        # inter_references_out = inter_references
        #
        # # 后两个只有two_stage的时候有值
        # if self.two_stage:
        #     # encoder的输出经过最后的分类头得到的输出
        #     # enc_outputs_class [bs,all hw,91]
        #     # encoder的输出经过最后的bbox头得到的输出
        #     # enc_outputs_coord_unact [bs,all hw,4]
        #     return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        #
        # # hs: 6层decoder输出 [n_decoder, bs, num_query, d_model] = [6, bs, 300, 256]
        # # init_reference_out: 初始化的参考点归一化中心坐标 [bs, 300, 2]
        # # inter_references: 6层decoder学习到的参考点归一化中心坐标  [6, bs, 300, 2]
        # #                   one-stage=[n_decoder, bs, num_query, 2]  two-stage=[n_decoder, bs, num_query, 4]
        # return hs, init_reference_out, inter_references_out, None, None
        # decoder

        hs_RGB, hs_T, inter_references_RGB, inter_references_T, Feature_RGB, Feature_T = self.decoder(tgt_RGB, tgt_T, reference_points_RGB, reference_points_T, memory_RGB, memory_T,
                                            spatial_shapes_RGB, spatial_shapes_T, level_start_index_RGB, level_start_index_T,
                                            valid_ratios_RGB, valid_ratios_T, query_embed_RGB, query_embed_T, mask_flatten_RGB, mask_flatten_T)

        inter_references_out_RGB = inter_references_RGB
        inter_references_out_T = inter_references_T
        if self.two_stage:
            return hs_RGB, hs_T, inter_references_out_RGB, inter_references_out_T, enc_outputs_class, enc_outputs_coord_unact
        return hs_RGB, hs_T, init_reference_out_RGB, init_reference_out_T, inter_references_out_RGB, inter_references_out_T, None, None, None, None, Feature_RGB, Feature_T


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8,
                 # Deformable的内容
                 n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn 的内容
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        # 一个全连接，一个激活，一个dropout，一个全连接
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        # add 和 dropout
        src = src + self.dropout3(src2)
        # norm
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        src: [bs,all hw,256]
        pos: [bs,all hw,256]
        reference_points: [bs,all hw,4,2]
        spatial_shapes: [4,2] 4个特征层的高宽
        level_start_index: 各个特征层的起始index的下标 如: [    0,  8056, 10070, 10583]
        padding_mask: [bs,all hw]

        src: 多尺度特征图(4个flatten后的特征图)  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        reference_points: 4个flatten后特征图对应的归一化参考点坐标 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        pos: 4个flatten后特征图对应的位置编码（多尺度位置编码） [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        spatial_shapes: 4个特征图的shape [4, 2]
        level_start_index: [4] 4个flatten后特征图对应被flatten后的起始索引  如[0,15100,18900,19850]
        padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        # self attention + add + norm
        # query = flatten后的多尺度特征图 + scale-level pos
        # key = 采样点  每个特征点对应周围的4个可学习的采样点
        # value = flatten后的多尺度特征图
        # with_pos_embed 就是将src和pos相加, src是上一层encoder的输出
        # 1. self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        # 2. add
        src = src + self.dropout1(src2)
        # 3. norm
        src = self.norm1(src)

        # 4. ffn + add & norm
        src = self.forward_ffn(src)

        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # 6层DeformableTransformerEncoderLayer
        self.layers = _get_clones(encoder_layer, num_layers)
        # 6
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        生成参考点   reference points  为什么参考点是中心点？  为什么要归一化？
        spatial_shapes: 4个特征图的shape [4, 2]
        valid_ratios: 4个特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        device: cuda:0
        """
        # spatial_shapes [特征层数,2] valid_ratios [bs,特征层数,2]
        reference_points_list = []

        # 遍历4个特征图的shape  比如 H_=100  W_=150
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # 生成网格点,从0.5开始 到 减掉一个0.5
            # 0.5 -> 99.5 取100个点  0.5 1.5 2.5 ... 99.5
            # 0.5 -> 149.5 取150个点 0.5 1.5 2.5 ... 149.5
            # ref_y: [100, 150]  第一行：150个0.5  第二行：150个1.5 ... 第100行：150个99.5
            # ref_x: [100, 150]  第一行：0.5 1.5...149.5   100行全部相同
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))

            # 坐标进行缩放 valid_ratios[:, None, lvl, 1] * H_是在H_基础上进一步缩减范围

            # reshape(-1) 拉平会变成一维的 shape=hw，[None]，会在最前面加上一个1维度 -> [1,hw] -> [2,hw]
            # [100, 150] -> [bs, 15000]  150个0.5 + 150个1.5 + ... + 150个99.5 -> 除以100 归一化
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            # [2,hw]
            # [100, 150] -> [bs, 15000]  100个: 0.5 1.5 ... 149.5  -> 除以150 归一化
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # [2,hw,2]
            # [bs, 15000, 2] 每一项都是xy
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # 所有特征层的参考点拼在一起 [bs,all hw,2]
        # list4: [bs, H/8*W/8, 2] + [bs, H/16*W/16, 2] + [bs, H/32*W/32, 2] + [bs, H/64*W/64, 2] ->
        # [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2]
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points[:,:,None] -> [2,all hw,1,2]
        # valid_ratios[:,None] -> [bs,1,特征层数量,2]
        # reference_points: [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 2] -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 1, 2]
        # valid_ratios: [1, 4, 2] -> [1, 1, 4, 2]
        # 复制4份 每个特征点都有4个归一化参考点 -> [bs, H/8*W/8+H/16*W/16+H/32*W/32+H/64*W/64, 4, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        # [2,all hw,4,2]
        # 4个flatten后特征图的归一化参考点坐标
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        src: [bs,all hw,256] backbone的特征
        spatial_shapes: [特征层的数量,2] 各个特征层的高宽
        level_start_index: 各个层的 all hw中的起始坐标位置
        valid_ratios: [bs,4,2] todo 不知道什么用
        pos: [bs,all hw,256] 位置编码
        padding_mask: [bs, all hw]

        src: 多尺度特征图(4个flatten后的特征图)  [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        spatial_shapes: 4个特征图的shape [4, 2]
        level_start_index: [4] 4个flatten后特征图对应被flatten后的起始索引  如[0,15100,18900,19850]
        valid_ratios: 4个特征图中非padding部分的边长占其边长的比例  [bs, 4, 2]  如全是1
        pos: 4个flatten后特征图对应的位置编码（多尺度位置编码） [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        padding_mask: 4个flatten后特征图的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        output = src
        # 获取参考点
        # encoder的参考点是grid生成的，并且不会精炼，不会有迭代的更新
        # [bs,all hw,4,2]
        # 4个flatten后特征图的归一化参考点坐标 每个特征点有4个参考点 xy坐标 [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 4, 2]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            # [bs,all hw,256]
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        # 经过6层encoder增强后的新特征  每一层不断学习特征层中每个位置和4个采样点的相关性，最终输出的特征是增强后的特征图
        # [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Deformable DETR实现的Attention
        # cross attention
        # self.cross_attn = MSDeformAttn_cross(d_model, n_levels, n_heads, n_points)
        self.cross_attn_inter = MSDeformAttn_cross_interactive(d_model, n_levels, n_heads, n_points)
        # self.cross_attn_base = MSDeformAttn_cross_base(d_model, n_levels, n_heads, n_points)

        self.dropout1_RGB = nn.Dropout(dropout)
        self.norm1_RGB = nn.LayerNorm(d_model)
        self.dropout1_T = nn.Dropout(dropout)
        self.norm1_T = nn.LayerNorm(d_model)

        # 标准的Attention
        # self attention
        self.self_attn_RGB = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2_RGB = nn.Dropout(dropout)
        self.norm2_RGB = nn.LayerNorm(d_model)

        self.self_attn_T = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2_T = nn.Dropout(dropout)
        self.norm2_T = nn.LayerNorm(d_model)

        # ffn
        self.linear1_RGB = nn.Linear(d_model, d_ffn)
        self.activation_RGB = _get_activation_fn(activation)
        self.dropout3_RGB = nn.Dropout(dropout)
        self.linear2_RGB = nn.Linear(d_ffn, d_model)
        self.dropout4_RGB = nn.Dropout(dropout)
        self.norm3_RGB = nn.LayerNorm(d_model)

        self.linear1_T = nn.Linear(d_model, d_ffn)
        self.activation_T = _get_activation_fn(activation)
        self.dropout3_T = nn.Dropout(dropout)
        self.linear2_T = nn.Linear(d_ffn, d_model)
        self.dropout4_T = nn.Dropout(dropout)
        self.norm3_T = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn_RGB(self, tgt):
        tgt2 = self.linear2_RGB(self.dropout3_RGB(self.activation_RGB(self.linear1_RGB(tgt))))
        tgt = tgt + self.dropout4_RGB(tgt2)
        tgt = self.norm3_RGB(tgt)
        return tgt

    def forward_ffn_T(self, tgt):
        tgt2 = self.linear2_T(self.dropout3_T(self.activation_T(self.linear1_T(tgt))))
        tgt = tgt + self.dropout4_T(tgt2)
        tgt = self.norm3_T(tgt)
        return tgt

    def forward(self, tgt_RGB, tgt_T, query_pos_RGB, query_pos_T, reference_points_RGB, reference_points_T,
                src_RGB, src_T, src_spatial_shapes_RGB, src_spatial_shapes_T, level_start_index_RGB, level_start_index_T,
                src_padding_mask_RGB=None, src_padding_mask_T=None):
        """
        tgt 是上一层的输出 [bs,300,256]
        query_pos 就是外面的query_embed [bs,300,256]
        reference_points 各个image在300个query上每个特征层上的参考点的坐标 [bs,300,4,2]
        src是encoder的输出 [bs,all hw,256]
        src_spatial_shapes 各个特征层的高宽 [4,2]
        level_start_index 各个特征层的起始下标
        src_padding_mask mask

        tgt: 预设的query embedding [bs, 300, 256]
        query_pos: 预设的query pos [bs, 300, 256]
        reference_points: query pos通过一个全连接层->2维  [bs, 300, 4, 2] = [bs, num_query, n_layer, 2]
                          iterative bounding box refinement时 = [bs, num_query, n_layer, 4]
        src: 第一层是encoder的输出memory 第2-6层都是上一层输出的output
        src_spatial_shapes: [4, 2] 4个特征层的原始shape
        src_level_start_index: [4,] 4个特征层flatten后的开始index
        src_padding_mask: 4个特征层flatten后的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """

        # self attention
        # tgt和query_pos相加, q k就是这两个构成了
        # query embedding  +  query_pos
        q_RGB = k_RGB = self.with_pos_embed(tgt_RGB, query_pos_RGB)
        q_T = k_T = self.with_pos_embed(tgt_T, query_pos_T)
        # 1. self-attention, q k在上面创建了，value就还是tgt
        # self-attention
        # 第一个attention的目的：学习各个物体之间的关系/位置   可以知道图像当中哪些位置会存在物体  物体信息->tgt
        # 所以qk都是query embedding + query pos   v就是query embedding
        tgt2_RGB = self.self_attn_RGB(q_RGB.transpose(0, 1), k_RGB.transpose(0, 1), tgt_RGB.transpose(0, 1))[0].transpose(0, 1)
        tgt2_T = self.self_attn_T(q_T.transpose(0, 1), k_T.transpose(0, 1), tgt_T.transpose(0, 1))[0].transpose(0, 1)
        # 2. add
        tgt_RGB = tgt_RGB + self.dropout2_RGB(tgt2_RGB)
        tgt_T = tgt_T + self.dropout2_T(tgt2_T)
        # 3. norm
        tgt_RGB = self.norm2_RGB(tgt_RGB)
        tgt_T = self.norm2_T(tgt_T)

        # Deformable DETR实现的attention
        # 4. cross attention

        # reference_points 各个image在300个query上每个特征层上的目标点位 [bs,300,4,2]
        # src是encoder的输出 [bs,all hw,256]
        # src_spatial_shapes 各个特征层的高宽 [4,2]
        # level_start_index 各个特征层的起始下标
        # src_padding_mask mask

        # cross attention  使用（多尺度）可变形注意力模块替代原生的Transformer交叉注意力
        # 第二个attention的目的：不断增强encoder的输出特征，将物体的信息不断加入encoder的输出特征中去，更好地表征了图像中的各个物体
        # 所以q=query embedding + query pos, k = query pos通过一个全连接层->2维, v=上一层输出的output
        # tgt_cat = torch.cat([tgt_RGB, tgt_T], dim=1)
        # query_pos = torch.cat([query_pos_RGB, query_pos_T], dim=1)
        # reference_points = torch.cat([reference_points_RGB, reference_points_T], dim=1)
        # src = torch.cat([src_RGB, src_T], dim=1)
        # src_spatial_shapes = torch.cat([src_spatial_shapes_RGB, src_spatial_shapes_T], dim=0)
        # level_start_index_T = (src_spatial_shapes_RGB[:,0]*src_spatial_shapes_RGB[:,1]).sum() + level_start_index_T
        # level_start_index = torch.cat([level_start_index_RGB, level_start_index_T], dim=0)
        # src_padding_mask = torch.cat([src_padding_mask_RGB, src_padding_mask_T], dim=1)

        # tgt2 = self.cross_attn(self.with_pos_embed(tgt_cat, query_pos),
        #                        reference_points,
        #                        src, src_spatial_shapes, level_start_index, src_padding_mask)

        tgt2_RGB, tgt2_T, Feature_RGB, Feature_T = self.cross_attn_inter(tgt_RGB, query_pos_RGB,
                                                tgt_T, query_pos_T,
                                                reference_points_RGB, reference_points_T,
                                                src_RGB, src_T, src_spatial_shapes_RGB, src_spatial_shapes_T,
                                                level_start_index_RGB, level_start_index_T, src_padding_mask_RGB,
                                                src_padding_mask_T)

        # tgt2_RGB, tgt2_T = self.cross_attn_base(self.with_pos_embed(tgt_RGB, query_pos_RGB), self.with_pos_embed(tgt_T, query_pos_T),
        #                        reference_points_RGB, reference_points_T,
        #                        src_RGB, src_T, src_spatial_shapes_RGB, src_spatial_shapes_T,
        #                        level_start_index_RGB, level_start_index_T, src_padding_mask_RGB, src_padding_mask_T)


        # 5. add
        # tgt = tgt_cat + self.dropout1(tgt2)
        tgt_RGB = tgt_T + self.dropout1_RGB(tgt2_RGB)
        tgt_T = tgt_RGB + self.dropout1_T(tgt2_T)
        # 6. norm
        tgt_RGB = self.norm1_RGB(tgt_RGB)
        tgt_T = self.norm1_T(tgt_T)

        # 7. ffn add & norm
        tgt_RGB = self.forward_ffn_RGB(tgt_RGB)
        tgt_T = self.forward_ffn_T(tgt_T)

        # tgt = torch.cat([tgt_RGB, tgt_T], dim=1)

        # [bs, 300, 256]  self-attention输出特征 + cross-attention输出特征
        # 最终的特征：知道图像中物体与物体之间的位置关系 + encoder增强后的图像特征 + 图像与物体之间的关系
        # return tgt
        return tgt_RGB, tgt_T, Feature_RGB, Feature_T


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        # 6层DeformableTransformerDecoderLayer
        self.layers = _get_clones(decoder_layer, num_layers)
        # 6
        self.num_layers = num_layers
        # True  默认是返回所有Decoder层输出 计算所有层损失
        self.return_intermediate = return_intermediate

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        # 策略1  iterative bounding box refinement
        self.bbox_embed = None
        # 策略2  two-stage Deformable DETR
        self.class_embed = None

    def forward(self, tgt_RGB, tgt_T, reference_points_RGB, reference_points_T, src_RGB, src_T,
                src_spatial_shapes_RGB, src_spatial_shapes_T, src_level_start_index_RGB, src_level_start_index_T,
                src_valid_ratios_RGB, src_valid_ratios_T, query_pos_RGB=None, query_pos_T=None,
                src_padding_mask_RGB=None, src_padding_mask_T=None):
        """
        tgt 从query_embed分出来的 [bs,300,256]
        reference_points reference_point query_embed经过全连接网络生成的 [bs,300,2], 参考点的初始参考点
        src encoder的输出 [bs, all hw,256]
        src_spatial_shapes 各个特征层的高宽 [4,2]
        src_level_start_index 各个特征层的起始下标
        src_valid_ratios [bs,4,2] 各个图像真实的高宽占据Mask大小中的比率
        query_pos 就是外面的query_embed [bs,300,256]
        src_padding_mask [bs,all hw]

        tgt: 预设的query embedding [bs, 300, 256]
        query_pos: 预设的query pos [bs, 300, 256]
        reference_points: query pos通过一个全连接层->2维  [bs, 300, 2]
        src: encoder最后的输出特征 即memory [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64, 256]
        src_spatial_shapes: [4, 2] 4个特征层的原始shape
        src_level_start_index: [4,] 4个特征层flatten后的开始index
        src_padding_mask: 4个特征层flatten后的mask [bs, H/8 * W/8 + H/16 * W/16 + H/32 * W/32 + H/64 * W/64]
        """
        output_RGB = tgt_RGB
        output_T = tgt_T

        # # 中间各层+首尾两层=6层输出的解码结果
        # intermediate = []
        # # 中间各层+首尾两层输出的参考点（不断矫正）
        # intermediate_reference_points = []
        intermediate_RGB = []
        intermediate_reference_points_RGB = []
        intermediate_T = []
        intermediate_reference_points_T = []
        Features_RGB, Features_T = [], []

        for lid, layer in enumerate(self.layers):
            # 得到参考点坐标
            # two stage
            if reference_points_RGB.shape[-1] == 4:
                # 不过现在代码中生成参考点的Linear输出就是维度就是2，并非设置，可能作者在实验中试验过=4的情况
                reference_points_input_RGB = reference_points_RGB[:, :, None] \
                                         * torch.cat([src_valid_ratios_RGB, src_valid_ratios_RGB], -1)[:, None]
            else:
                # 参考点的坐标
                # one stage模式下参考点是query pos通过一个全连接层线性变化为2维的  中心坐标形式(x,y)
                assert reference_points_RGB.shape[-1] == 2
                # [bs,300,2] -> [bs,300,1,2] [bs,4,2] -> [bs,1,4,2]
                # reference_points_input [bs,300,4,2]
                # 参考点坐标也要按比例的进行缩放
                # [bs, 300, 1, 2] * [bs, 1, 4, 2] -> [bs, 300, 4, 2]=[bs, n_query, n_lvl, 2]
                reference_points_input_RGB = reference_points_RGB[:, :, None] * src_valid_ratios_RGB[:, None]
                reference_points_input_T = reference_points_T[:, :, None] * src_valid_ratios_T[:, None]
                # src_valid_ratios_All=torch.cat((src_valid_ratios_RGB,src_valid_ratios_T),dim=1)
                # reference_points_input_RGB = reference_points_RGB[:, :, None] * src_valid_ratios_All[:, None]
                # reference_points_input_T = reference_points_T[:, :, None] * src_valid_ratios_All[:, None]

            # 如果没有使用bbox精炼，那么每次的reference_points_input 其实是相同的
            # 如果使用了bbox精炼，那么每次的reference_points_input 是不同的，会使用decoder的输出得到偏移量修正

            # output 是上一层的输出
            # query_pos 就是外面的query_embed
            # reference_points_input 各个image在300个query上每个特征层上的目标点位
            # src是encoder的输出
            # src_spatial_shapes 各个特征层的高宽
            # src_level_start_index 各个特征层的起始下标
            # src_padding_mask mask

            # decoder layer
            # output: [bs, 300, 256] = self-attention输出特征 + cross-attention输出特征
            # 知道图像中物体与物体之间的关系 + encoder增强后的图像特征 + 图像与物体之间的关系
            # output = layer(output_RGB, output_T, query_pos_RGB, query_pos_T, reference_points_input_RGB, reference_points_input_T,
            #                src_RGB, src_T, src_spatial_shapes_RGB, src_spatial_shapes_T, src_level_start_index_RGB, src_level_start_index_T,
            #                src_padding_mask_RGB, src_padding_mask_T)
            output_RGB, output_T, Feature_RGB, Feature_T = layer(output_RGB, output_T, query_pos_RGB, query_pos_T, reference_points_input_RGB,
                                         reference_points_input_T,
                                         src_RGB, src_T, src_spatial_shapes_RGB, src_spatial_shapes_T,
                                         src_level_start_index_RGB, src_level_start_index_T,
                                         src_padding_mask_RGB, src_padding_mask_T)

            # 如果使用了with_box_refine模式, 这个地方的bbox_embed 是非None的
            # hack implementation for iterative bounding box refinement

            # hack implementation for iterative bounding box refinement
            # 使用iterative bounding box refinement 这里的self.bbox_embed就不是None
            # 如果没有iterative bounding box refinement那么reference_points是不变的
            # 每层参考点都会根据上一层的输出结果进行矫正
            # if self.bbox_embed is not None:
            #     # output 是上一层decoder的输出，经过bbox预测网络 [bs,300,4]
            #     # 得到此次的bbox的偏移量修正
            #     # [bs, 300, 256] -> [bs, 300, 4(xywh)]
            #     # tmp = self.bbox_embed[lid](output)
            #     # two stage
            #     if reference_points.shape[-1] == 4:
            #         # inverse_sigmoid sigmoid的反函数
            #         new_reference_points = tmp + inverse_sigmoid(reference_points)
            #         new_reference_points = new_reference_points.sigmoid()
            #     else:  # one stage
            #         # [bs,300,2]
            #         assert reference_points.shape[-1] == 2
            #         # [bs,300,4]
            #         new_reference_points = tmp
            #         # 前两个是bbox的中心坐标，中心坐标用reference_points的内容+偏移量修正
            #         # 根据decoder每层解码的特征图->回归头（不共享参数） 得到相对参考点的偏移量xy
            #         # 然后再加上参考点坐标（反归一化），再进行sigmoid归一化 得到矫正的参考点
            #         new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
            #         # [bs,300,4] 经过sigmoid 约束在0-1
            #         new_reference_points = new_reference_points.sigmoid()
            #     # 替换了原来的reference_points
            #     # reference_points: [bs, 300, 2] -> [bs, 300, 4]
            #     # .detach() 取消了梯度  因为这个参考点在各层相当于作为先验的角色
            #     reference_points = new_reference_points.detach()
            if self.bbox_embed is not None:
                tmp_RGB = self.bbox_embed[lid](output_RGB)
                tmp_T = self.bbox_embed[lid](output_T)
                if reference_points_RGB.shape[-1] == 4:
                    new_reference_points = tmp_RGB + inverse_sigmoid(reference_points_RGB)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points_RGB.shape[-1] == 2
                    new_reference_points_RGB = tmp_RGB
                    new_reference_points_T = tmp_T
                    new_reference_points_RGB[..., :2] = tmp_RGB[..., :2] + inverse_sigmoid(reference_points_RGB)
                    new_reference_points_T[..., :2] = tmp_T[..., :2] + inverse_sigmoid(reference_points_T)
                    new_reference_points_RGB = new_reference_points_RGB.sigmoid()
                    new_reference_points_T = new_reference_points_T.sigmoid()
                reference_points_RGB = new_reference_points_RGB.detach()
                reference_points_T = new_reference_points_T.detach()

            if self.return_intermediate:
                intermediate_RGB.append(output_RGB)
                intermediate_T.append(output_T)
                intermediate_reference_points_RGB.append(reference_points_RGB)
                intermediate_reference_points_T.append(reference_points_T)


            # # 返回前几层，否则只返回最后一层
            # # 默认返回6个decoder层输出一起计算损失
            # if self.return_intermediate:
            #     intermediate.append(output)
            #     # 如果是精炼模式，reference_points在各个层之后是不同的
            #     # 如果不是精炼模型，reference_points在各个层之后还是同一份内容
            #
            #     # 0 [6, bs, 300, 256] 6层decoder输出
            #     # 1 [6, bs, 300, 2] 6层decoder的参考点归一化中心坐标  一般6层是相同的
            #     # 但是如果是iterative bounding box refinement会不断学习迭代得到新的参考点 6层一半不同
            #     reference_points = torch.cat([reference_points_RGB, reference_points_T], dim=1)
            #     intermediate_reference_points.append(reference_points)
            Features_RGB.append(Feature_RGB)
            Features_T.append(Feature_T)

        # if self.return_intermediate:
        #     return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        #
        # return output, reference_points
        if self.return_intermediate:
            return torch.stack(intermediate_RGB), torch.stack(intermediate_T), torch.stack(intermediate_reference_points_RGB), torch.stack(intermediate_reference_points_T), Features_RGB, Features_T

        return output_RGB, output_T, reference_points_RGB, reference_points_T, Features_RGB, Features_T


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        # 256
        d_model=args.hidden_dim,
        # 头的数量 8
        nhead=args.nheads,
        # encoder的层数 6
        num_encoder_layers=args.enc_layers,
        # decoder的层数
        num_decoder_layers=args.dec_layers,
        # 1024
        dim_feedforward=args.dim_feedforward,
        # 0.1
        dropout=args.dropout,
        # 激活函数
        activation="relu",
        # 返回decoder的中间层的输出
        return_intermediate_dec=True,
        # 4
        num_feature_levels=args.num_feature_levels,
        # 4
        dec_n_points=args.dec_n_points,
        # 4
        enc_n_points=args.enc_n_points,
        # 双阶段方式
        two_stage=args.two_stage,
        # 300
        two_stage_num_proposals=args.num_queries)


