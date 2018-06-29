# -*- coding:utf-8 -*-

import argparse
import os
import random
from PIL import Image
from tqdm import tqdm

SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/original', help='hehe')
parser.add_argument('--output_dir', default='data/64x64_data', help='hehe')


def resize_and_save(filename, output_dir, category, size=SIZE):
    img = Image.open(filename)
    img = img.resize((size, size), Image.BILINEAR)
    img = img.convert('L')
    output_name = '_'.join([str(category), os.path.basename(filename)])
    img.save(os.path.join(output_dir, output_name))


def split_data(original_dir, train_dir, split=0.8):
    cnt = 0
    category_dict = {}
    for c in os.listdir(original_dir):
        category_dict[cnt] = c
        cnt += 1
    files_dict = {}
    for k, v in category_dict.items():
        category_dir = '/'.join([original_dir, v])
        files = os.listdir(category_dir)
        files = [os.path.join(category_dir, f) for f in files]
        files_dict[k] = files

    random.seed(231)
    train_filenames = {}
    val_filenames = {}
    for k, v in files_dict.items():
        v.sort()
        random.shuffle(v)
        split_num = int(split * len(v))
        train_filenames[k] = v[: split_num]
        val_filenames[k] = v[split_num:]

    return train_filenames, val_filenames, category_dict


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "no dataset at {}".format(args.data_dir)

    train_dir = os.path.join(args.data_dir, 'train')
    train_dir.replace('\\', '/')

    train_filenames, val_filenames, category_dict = split_data(args.data_dir, train_dir)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("output dir {} already exists".format(args.output_dir))

    filenames = {'train': train_filenames,
                 'val': val_filenames}

    for split in ['train', 'val']:
        output_dir_split = os.path.join(args.output_dir, '{}'.format(split))
        output_dir_split.replace('\\', '/')
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print('hehehe')

        for ca_k, files in filenames[split].items():
            print("Processing {} data, {} category, saving preprocessed data to {}".
                  format(split, category_dict[ca_k], output_dir_split))
            for fname in tqdm(files):
                resize_and_save(fname, output_dir_split, category_dict[ca_k])


