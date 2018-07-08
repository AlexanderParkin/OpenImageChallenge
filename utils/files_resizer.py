# -*- coding: utf-8 -*-
"""
Alexander Parkin
"""
import os
from tqdm import tqdm
import glob
import argparse
from scipy.misc import imread, imresize, imsave
from multiprocessing import Pool

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def resize_img(image, min_size=-1):

    height, width = image.shape[0], image.shape[1]

    scale_ratio = min_size / max(width, height)

    if min_size == -1 or scale_ratio >= 1:
        return image

    return imresize(image, scale_ratio,interp='bilinear')

def worker(args):
    min_size, filepath = args
    if not os.path.isfile(filepath):
        return
    try:
        img = imread(filepath)
        if min_size > max(img.shape):
            return
        img = resize_img(img, min_size)
        if len(img.shape) == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        imsave(filepath, img)

    except Exception as e:
        print ('Error ' + filepath)
        print (e)
        if os.path.getsize(filepath) == 0:
            os.remove(filepath)
        return

def main(args):

    argss = glob.glob(os.path.join(args.save_dir + '*.jpg'))
    argss = [(args.min_size, x) for x in argss]
    with Pool(args.num_workers) as p:
        for _ in tqdm(p.imap(worker, argss), total = len(argss), desc = 'Processes'):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for training student model')

    parser.add_argument('--save_dir',
                        required=True,
                        type=str, 
                        help='save dir path')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='number of workers')
    parser.add_argument('--min_size',
                        type=int,
                        help='max image',
                        default=-1)
    args = parser.parse_args()
    main(args)
