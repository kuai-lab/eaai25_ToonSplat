#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Render test views, and report the metrics
'''

import os
import argparse
from PIL import Image
import imageio
import torch
import numpy as np
import pdb

import skimage
from skimage.metrics import structural_similarity as ssim
import lpips

loss_fn_alex = lpips.LPIPS(net='alex')

def np_img_to_torch_img(np_img):
    """convert a numpy image to torch image
    numpy use Height x Width x Channels
    torch use Channels x Height x Width
    Arguments:
        np_img {[type]} -- [description]
    """
    assert isinstance(np_img, np.ndarray), f'cannot process data type: {type(np_img)}'
    if len(np_img.shape) == 4 and (np_img.shape[3] == 3 or np_img.shape[3] == 1):
        return torch.from_numpy(np.transpose(np_img, (0, 3, 1, 2)))
    if len(np_img.shape) == 3 and (np_img.shape[2] == 3 or np_img.shape[2] == 1):
        return torch.from_numpy(np.transpose(np_img, (2, 0, 1)))
    elif len(np_img.shape) == 2:
        return torch.from_numpy(np_img)
    else:
        raise ValueError(f'cannot process this image with shape: {np_img.shape}')

def eval_metrics(gts, preds):
    results = {
        'ssim': [],
        'psnr': [],
        'lpips': []
    }
    for gt, pred in zip(gts, preds):
        results['ssim'].append(ssim(pred, gt, multichannel=True))
        results['psnr'].append(skimage.metrics.peak_signal_noise_ratio(gt, pred))
        results['lpips'].append(
            float(loss_fn_alex(np_img_to_torch_img(pred[None])/127.5-1, np_img_to_torch_img(gt[None])/127.5-1)[0, 0, 0, 0].data)
        )
    for k, v in results.items():
        results[k] = np.mean(v)
    return results

def read_images_from_folder(folder):
    dic = {}
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            image = imageio.imread(image_path)
            dic[filename] = image

    return [value for key, value in sorted(dic.items())]

def read_images_from_folder_PIL(folder):
    dic = {}
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            image = Image.open(image_path)
            if image.size[0] >= image.size[1]:
                size = (1280,720)
            else: size = (720, 1280)
            image = image.resize(size)
            dic[filename] = np.array(image)
    # print(sorted(dic.items()))

    return [value for key, value in sorted(dic.items())]

def save_dict_to_txt(dictionary, filename):
    with open(filename, 'w') as f:
        for key, value in dictionary.items():
            f.write(f"{key}: {value}\n")

def main(opt):

    rgb_gt = read_images_from_folder_PIL(opt.rgb_gt_dir)
    rgb_pred = read_images_from_folder_PIL(opt.rgb_pred_dir)
    out_metrics = eval_metrics(rgb_gt, rgb_pred)
    print(out_metrics)
    save_dict_to_txt(out_metrics, os.path.join(opt.rgb_pred_dir, 'results_skl.txt'))
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_gt_dir', default='segmentations', type=str, help='rgb gt folder')
    parser.add_argument('--rgb_pred_dir', default='segmentations', type=str, help='rgb pred folder')

    opt = parser.parse_args()
    main(opt)

