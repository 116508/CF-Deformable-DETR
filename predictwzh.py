import cv2
from PIL import Image
import numpy as np
import os
import time

import torch
import datasets.transforms as T

from main import get_args_parser as get_main_args_parser
from models import build_model

torch.set_grad_enabled(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))

# 图像数据处理

def make_coco_transforms(image_set):

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

mytransform = make_coco_transforms('val')

# plot box by opencv
def plot_result(pil_img, prob, boxes, save_name=None, imshow=True, imwrite=False):
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    LABEL = 'person'
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL, round(p * 100, 2))

        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) - 10, int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    if imwrite:
        if not os.path.exists("./result/pred"):
            os.makedirs('./result/pred')
        cv2.imwrite('./result/pred/{}'.format(save_name), opencvImage)


# 将xywh转xyxy
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


def load_model(model_path, args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path)  # <-----------修改加载模型的路径
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model

from models.deformable_detr import PostProcess
# 图像的推断
def detect(RGB_im, T_im, model, prob_threshold=0.1):
    w,h=RGB_im.size
    t_shape=torch.tensor((h,w)).unsqueeze(0).to(device)
    RGB_im,T_im,target ,target_T=mytransform(RGB_im, T_im, None, None)
    RGB_img = RGB_im.unsqueeze(0)
    RGB_img = RGB_img.to(device)
    T_img = T_im.unsqueeze(0)
    T_img = T_img.to(device)
    img_all = torch.cat((RGB_img, T_img), dim=1)
    start = time.time()
    outputs_RGB, outputs_T = model(img_all)
    postProcess=PostProcess()
    results_RGB=postProcess(outputs_RGB,t_shape)[0]
    results_T=postProcess(outputs_T,t_shape)[0]
    probas_RGB = results_RGB['scores']
    lables_RGB = results_RGB['labels']
    keep_RGB = probas_RGB.cpu().detach().numpy() > prob_threshold
    keep_RGB_label=lables_RGB.cpu().detach().numpy()>0
    keep_RGB=np.logical_and(keep_RGB,keep_RGB_label)
    # keep_RGB = probas_RGB.max(-1).values > prob_threshold
    # print(outputs_T['pred_logits'].softmax(-1)[0, :, :-1])
    probas_T = results_T['scores']

    keep_T = probas_T.cpu().detach().numpy() > prob_threshold
    # keep_T = probas_T.max(-1).values > prob_threshold

    probas_RGB = probas_RGB.cpu().detach().numpy()
    # keep_RGB = keep_RGB.cpu().detach().numpy()
    probas_T = probas_T.cpu().detach().numpy()
    # keep_T = keep_T.cpu().detach().numpy()

    # convert boxes from [0; 1] to image scales
    bboxes_scaled_RGB =results_RGB['boxes'][keep_RGB]
    bboxes_scaled_T = results_RGB['boxes'][keep_T]
    end = time.time()
    return  probas_RGB[keep_RGB], bboxes_scaled_RGB, probas_T[keep_T], bboxes_scaled_T, end - start


if __name__ == "__main__":

    main_args = get_main_args_parser().parse_args()
    # 加载模型
    dfdetr = load_model('./exps/checkpoint0089.pth', main_args)  # <--修改为自己加载模型的路径

    files = "E:/experiment/RGBT_detection/DDetr-Single-mosaic/data/FLIR_shift/RGB_shift/val_45"  # <--修改为待预测图片所在文件夹路径
    RGB_path = files + '/visible/'
    T_path = files + '/ir/'
    RGB_names = os.listdir(RGB_path)

    cn = 0
    waste = 0
    for file in RGB_names:
        RGB_img_path = os.path.join(RGB_path, file)  # <--修改为待预测图片所在文件夹路径
        RGB_im = Image.open(RGB_img_path)
        # T_img_path = T_path + file[:-11] + 'lwir.png'  # <--修改为待预测图片所在文件夹路径
        T_img_path = T_path + file[:-7] + 'PreviewData.jpeg'
        T_im = Image.open(T_img_path)
        # FLIR_08866_PreviewData

        scores_RGB, boxes_RGB, scores_T, boxes_T, waste_time = detect(RGB_im, T_im, dfdetr, 0)

        plot_result(RGB_im, scores_RGB, boxes_RGB, save_name=file, imshow=True, imwrite=False)
        plot_result(T_im, scores_T, boxes_T, save_name=file, imshow=True, imwrite=False)

        # image_name_save, png = file.split('.')
        # image_name_save = image_name_save[:-8]  # lwir
        # result_path = './result'
        # RGB_image_path = os.path.join(result_path, 'visible/')
        # RGB_image_file = os.path.join(result_path, 'visible/' + image_name_save + '.txt')
        # T_image_path = os.path.join(result_path, 'lwir/')
        # T_image_file = os.path.join(result_path, 'lwir/' + image_name_save + '.txt')
        #
        # if not os.path.exists(RGB_image_path and T_image_path):
        #     os.makedirs(RGB_image_path)
        #     os.makedirs(T_image_path)

        # RGB_list_file = open(RGB_image_file, 'w')
        # for i in range(len(boxes_RGB)):
        #     image_write_txt = 'person' + ' ' + str(float('%.4f'% boxes_RGB[i][0])) + ' ' + str(float('%.4f'% boxes_RGB[i][1])) + ' ' \
        #                           + str(float('%.4f'% boxes_RGB[i][2])) + ' ' + str(float('%.4f'% boxes_RGB[i][3])) + ' ' + str(round(float(scores_RGB[i]), 8))
        #     RGB_list_file.write(image_write_txt)
        #     RGB_list_file.write('\n')
        # RGB_list_file.close()
        #
        # T_list_file = open(T_image_file, 'w')
        # for i in range(len(boxes_T)):
        #     image_write_txt = 'person' + ' ' + str(float('%.4f'% boxes_T[i][0])) + ' ' + str(float('%.4f'% boxes_T[i][1])) + ' ' \
        #                       + str(float('%.4f'% boxes_T[i][2])) + ' ' + str(float('%.4f'% boxes_T[i][3])) + ' ' + str(round(float(scores_T[i]), 8))
        #     T_list_file.write(image_write_txt)
        #     T_list_file.write('\n')
        # T_list_file.close()

        print("{} [INFO] {} time: {} done!!!".format(cn, file, waste_time))

        cn += 1
        waste += waste_time
    waste_avg = waste / cn
    print(waste_avg)
