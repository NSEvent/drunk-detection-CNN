#!/usr/bin/env python3

import argparse
import numpy as np
from PIL import Image


class DrunkDetector:
    def __init__(self, args):
        self.train_files = args.train_files

    def read_images(self):
        for file_name in self.train_files:
            thermal_data = np.zeros((128, 160))
            with Image.open(file_name) as img:
                for i in range(img.n_frames):
                    img.seek(i)
                    frame_data = np.array(img)
                    min_val = np.amin(frame_data)
                    frame_data -= min_val
                    for j in range(img.height):
                        for k in range(img.width):
                            thermal_data[j, k] += frame_data[j, k]
            print(file_name)
            print(thermal_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, help='[data|train|predict]')
    parser.add_argument('--train-files', nargs='+', type=str,
        help='Files to train model on.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dd = DrunkDetector(args)

    if args.mode == 'data':
        dd.read_images()


