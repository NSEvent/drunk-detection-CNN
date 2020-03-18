#!/usr/bin/env python3

# e.g.: python src/__main__.py data --train-files data/train/*/* --test-files data/test/*/* --val-files data/validation/*/*

import argparse
import datetime
import numpy as np
import os
import pickle
from PIL import Image
import re
from sklearn.svm import SVC


class DrunkDetector:
    def __init__(self, args):
        self.face_filename_pattern = '[0-4][0-9]_[a-z]*_[0-4]_f_[FM]_[0-9_]*\.tif'
        self.sober_filename_patter = '[0-4][0-9]_[a-z]*_1_f_[FM]_[0-9_]*\.tif'
        self.args = args
        self.data = {}

    def format_data(self):
        self.read_images(self.args.train_files, 'train')
        self.read_images(self.args.test_files, 'test')
        self.read_images(self.args.val_files, 'val')
        curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = '{}_{}'.format('data', curr_datetime)
        with open(os.path.join(self.args.output_dir, filename), 'wb') as out_file:
            pickle.dump(self.data, out_file)

    def read_images(self, files, split_set):
        for filename in files:
            basename = os.path.basename(filename)
            if not self.is_face_image(basename):
                continue

            with Image.open(filename) as img:
                self.data[split_set] = {
                    'thermal_sum': np.zeros((128, 160)),
                    'thermal_frames': np.zeros((img.n_frames, 128, 160)),
                    'y': self.is_sober_image(basename)
                }

                for i in range(img.n_frames):
                    img.seek(i)
                    frame_data = np.array(img)
                    min_val = np.amin(frame_data)
                    frame_data -= min_val
                    for j in range(img.height):
                        for k in range(img.width):
                            self.data[split_set]['thermal_sum'][j, k] += frame_data[j, k]

                    self.data[split_set]['thermal_frames'][i] = frame_data

    def is_face_image(self, filename):
        return re.match(self.face_filename_pattern, filename) is not None

    def is_sober_image(self, filename):
        return re.match(self.sober_filename_patter, filename) is not None

    def train(self):
        self.train_svm()

    def train_svm(self):
        svc = SVC()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='[data|train|predict]')
    parser.add_argument('-m', '--data-mode', type=str, help='[frames|sum]', default='frames')
    parser.add_argument('-o', '--output-dir', type=str, default=os.getcwd())
    parser.add_argument('-d', '--data', type=str, help='File containing data (from data mode).')
    parser.add_argument('--train-files', nargs='+', type=str, default=[],
        help='Files to train model on.')
    parser.add_argument('--test-files', nargs='+', type=str, default=[],
        help='Files to test model on.')
    parser.add_argument('--val-files', nargs='+', type=str, default=[],
        help='Files to validate model on.')
    args = parser.parse_args()
    if args.mode not in ['data', 'train', 'predict']:
        parser.print_help()
    return args


if __name__ == '__main__':
    args = parse_args()
    dd = DrunkDetector(args)

    if args.mode == 'data':
        dd.format_data()

    elif args.mode == 'train':
        dd.train()

    elif args.mode == 'predict':
        pass


