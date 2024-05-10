
import warnings


warnings.filterwarnings("ignore")
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from torch.nn.functional import linear, softmax
import os
import cv2

torch.set_grad_enabled(False)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# COCO classes
CLASSES = ['person']
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# ----------------------------------------------1. 加载模型及获取训练好的参数---------------------------------------------------
# 加载线上的模型
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
from datasets import build_dataset, get_coco_api_from_dataset
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
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    # 40个epoch后学习率下降
    parser.add_argument('--lr_drop', default=40, type=int)
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
    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')

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
    parser.add_argument('--coco_path', default='./data/kaist_clean_coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='./exps/r50_deformable_detr',
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

def rescale_sample(points, size):
    img_w, img_h = size
    b = points[0,:,:,:,:,:2]
    b = b * torch.tensor([img_w, img_h], dtype=torch.float32)
    b_x=torch.clamp(b[...,0], 0, img_w).unsqueeze(-1)
    b_y= torch.clamp(b[..., 1], 0, img_h).unsqueeze(-1)

    b=torch.cat([b_x,b_y],4)
    return b
def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    checkpoint=torch.load('./exps/checkpoint0019.pth' )
    model.load_state_dict( checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    # 获取训练好的参数

    # data_path = 'E:/experiment/RGBT_detection/Deformable-DETR/data/kaist_clean_coco/kaist_test'
    # RGB_path = data_path + '/visible/'
    # T_path = data_path + '/lwir/'
    # RGB_names = os.listdir(RGB_path)
    #
    # for i in range(len(RGB_names)):
    #     RGB_im = Image.open(RGB_path + RGB_names[i])
    #     RGB_img = transform(RGB_im).unsqueeze(0)
    #     RGB_img = RGB_img.to(device)
    #     T_name = RGB_names[i][:18] + 'lwir.png'
    #     T_im = Image.open(T_path + T_name)
    #     T_img = transform(T_im).unsqueeze(0)
    #     T_img = T_img.to(device)
    #     imgall = torch.cat((RGB_img, T_img), dim=1)
    #
    #     start_time = time.time()
    #     outputs, outputs_T = model(imgall)
    #
    #     probas_RGB = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    #     keep_RGB = probas_RGB.max(-1).values > 0.8
    #     bboxes_scaled_RGB = rescale_bboxes(outputs['pred_boxes'][0, keep_RGB].cpu(), RGB_im.size).numpy()
    #
    #     probas_T = outputs_T['pred_logits'].softmax(-1)[0, :, :-1]
    #     keep_T = probas_T.max(-1).values > 0.8
    #     bboxes_scaled_T = rescale_bboxes(outputs_T['pred_boxes'][0, keep_T].cpu(), T_im.size).numpy()
    #
    #     print('Test time: %.4f s' % (time.time() - start_time))
    #
    #     image_name_save, png = RGB_names[i].split('.')
    #     image_name_save = image_name_save[:-8]  # lwir
    #
    #     result_path = './result'
    #     RGB_image_path = os.path.join(result_path, 'visible/')
    #     RGB_image_file = os.path.join(result_path, 'visible/' + image_name_save + '.txt')
    #     T_image_path = os.path.join(result_path, 'lwir/')
    #     T_image_file = os.path.join(result_path, 'lwir/' + image_name_save + '.txt')
    #
    #     if not os.path.exists(RGB_image_path and T_image_path):
    #         os.makedirs(RGB_image_path)
    #         os.makedirs(T_image_path)
    #
    #     RGB_list_file = open(RGB_image_file, 'w')
    #     for i in range(len(bboxes_scaled_RGB)):
    #         image_write_txt = 'person' + ' ' + str(np.round(bboxes_scaled_RGB[i][0], 4)) + ' ' + str(np.round(bboxes_scaled_RGB[i][1], 4)) + ' ' \
    #                           + str(np.round(bboxes_scaled_RGB[i][2], 4)) + ' ' + str(np.round(bboxes_scaled_RGB[i][3], 4)) + ' ' + str(
    #             round(float(keep_RGB[i]), 8))
    #         RGB_list_file.write(image_write_txt)
    #         RGB_list_file.write('\n')
    #     RGB_list_file.close()
    #
    #     T_list_file = open(T_image_file, 'w')
    #     for i in range(len(bboxes_scaled_T)):
    #         image_write_txt = 'person' + ' ' + str(np.round(bboxes_scaled_T[i][0], 4)) + ' ' + str(np.round(bboxes_scaled_T[i][1], 4)) + ' ' \
    #                           + str(np.round(bboxes_scaled_T[i][2], 4)) + ' ' + str(np.round(bboxes_scaled_T[i][3], 4)) + ' ' + str(
    #             round(float(keep_T[i]), 8))
    #         T_list_file.write(image_write_txt)
    #         T_list_file.write('\n')
    #     T_list_file.close()






    # --------------------------------------------2.下载图像并进行预处理和前馈过程--------------------------------------------------
    # 线上下载图像
    url = 'E:\\experiment\\RGBT_detection\\Deformable-DETR\\data\\kaist_clean_coco\\kaist_test\\visible\\set06_V000_I00019_visible.png'
    im = Image.open(url)
    # img_path = '/home/wujian/000000039769.jpg'
    # im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)
    img = img.to(device)
    url2 = 'E:\\experiment\\RGBT_detection\\Deformable-DETR\\data\\kaist_clean_coco\\kaist_test\\lwir\\set06_V000_I00019_lwir.png'
    im2 = Image.open(url2)
    # img_path = '/home/wujian/000000039769.jpg'
    # im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img2 = transform(im2).unsqueeze(0)
    img2 = img2.to(device)
    imgall=torch.cat((img,img2),dim=1)
    # propagate through the model
    outputs, outputs_T= model(imgall)

    # keep only predictions with 0.7+ confidence
    temp = outputs['pred_logits'].softmax(-1)
    probas=temp[0, :, :]

    keep = probas.max(-1).values > 0.8

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), im.size)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(22, 28))  # [11,2]
    # 可视化


    # 可视化框和类别

    ax.imshow(im)
    for(xmin,ymin,xmax,ymax) in bboxes_scaled:
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='blue', linewidth=1))
        # ax.axis('off')
        # ax.set_title(CLASSES[probas[idx].argmax()], fontsize=30)

    fig.tight_layout()  # 自动调整子图来使其填充整个画布
    plt.savefig("border.eps", dpi=100, format="eps")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)