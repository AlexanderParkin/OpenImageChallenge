# -*- coding: utf-8 -*-
"""
Alexander Parkin
"""
import os
from time import time
from tqdm import tqdm
import pandas as pd
import urllib.request
import argparse
from scipy.misc import imread, imresize, imsave
from multiprocessing import Pool
from PIL import Image

def resize_img(image, min_size=-1):

    width, height = image.size

    scale_ratio = min_size / max(width, height)

    if min_size == -1 or scale_ratio >= 1:
        return image

    return image.resize((int(width * scale_ratio), int(height * scale_ratio)),
                        Image.ANTIALIAS)

def worker(args):
    save_dir, min_size, (_, row) = args
    url = row.OriginalURL
    filepath = '___'.join([row.ImageID, url.split('/')[-1]])
    filepath = os.path.join(save_dir, filepath)
    if os.path.isfile(filepath):
        return
    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception:
        print ('Download error ' + filepath)
    try:
        img = Image.open(filepath)
        if img.mode == 'CMYK' or img.mode == 'RGBA':
            img = img.convert('RGB')
        if max(img.size) > min_size:
            img = resize_img(img, min_size)
            img.save(filepath)
    except Exception as e:
        print ('Resize error ' + filepath)
        print (e)
    return

def main(args):
    save_dir = args.save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    df = pd.read_csv(args.datalist_path)
    delete_indexes = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        filepath = '___'.join([row.ImageID, row.OriginalURL.split('/')[-1]])
        filepath = os.path.join(save_dir, filepath)
        if os.path.isfile(filepath):
            delete_indexes.append(idx)
    df = df.drop(delete_indexes)

    argss = [(save_dir, args.min_size, row) for row in df.iterrows()]
    with Pool(args.num_workers) as p:
        for _ in tqdm(p.imap(worker, argss), total = len(argss), desc = 'Processes'):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for training student model')

    parser.add_argument('--save_dir',
                        required=True,
                        type=str, 
                        help='save dir path')

    parser.add_argument('--datalist_path',
                        required=True,
                        type=str, 
                        help='img datalist path')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='number of workers')
    parser.add_argument('--min_size',
                        type=int,
                        help='max image shape will be min size',
                        default=-1)
    args = parser.parse_args()
    main(args)
