#!/usr/bin/env python3

# e.g.: python drunk_detector data --train-files data/train/*/* --test-files data/test/*/* --val-files data/validation/*/*

import argparse
import datetime
import numpy as np
import os
import pickle
from PIL import Image
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Model types
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class DrunkDetector:
    def __init__(self, args):
        self.face_filename_pattern = '[0-4][0-9]_[a-z]*_[0-4]_f_[FM]_[0-9_]*\.tif'
        self.sober_filename_patter = '[0-4][0-9]_[a-z]*_1_f_[FM]_[0-9_]*\.tif'
        self.args = args
        self.data = { 'train': [], 'test': [], 'val': [] }

    def format_data(self):
        self.read_images(self.args.train_files, 'train')
        self.read_images(self.args.test_files, 'test')
        self.read_images(self.args.val_files, 'val')
        curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = '{}_{}.pickle'.format('data', curr_datetime)
        with open(os.path.join(self.args.output_dir, filename), 'wb') as out_file:
            pickle.dump(self.data, out_file)

    def read_images(self, files, split_set):
        for filename in files:
            basename = os.path.basename(filename)
            if not self.is_face_image(basename):
                continue

            with Image.open(filename) as img:
                # breakpoint()
                self.data[split_set].append({
                    'filename': basename,
                    'thermal_sum': np.zeros((128, 160)),
                    'thermal_frames': np.zeros((img.n_frames, 128, 160)),
                    'y': self.is_sober_image(basename)
                })

                for i in range(img.n_frames):
                    img.seek(i)
                    frame_data = np.array(img)
                    min_val = np.amin(frame_data)
                    frame_data -= min_val
                    for j in range(img.height):
                        for k in range(img.width):
                            self.data[split_set][-1]['thermal_sum'][j, k] += frame_data[j, k]

                    self.data[split_set][-1]['thermal_frames'][i] = frame_data

    def is_face_image(self, filename):
        return re.match(self.face_filename_pattern, filename) is not None

    def is_sober_image(self, filename):
        return re.match(self.sober_filename_patter, filename) is not None

    def train(self):
        if self.args.data is None:
            print('Error: No data given.')
            exit(1)

        with open(self.args.data, 'rb') as data_file:
            self.data = pickle.load(data_file)

        if self.args.data_mode == 'frames':
            self.train_X = np.array([datum['thermal_frames'] for datum in self.data['train']])
            self.val_X = np.array([datum['thermal_frames'] for datum in self.data['val']])
            self.test_X = np.array([datum['thermal_frames'] for datum in self.data['test']])
        elif self.args.data_mode == 'sum':
            self.train_X = np.array([datum['thermal_sum'] for datum in self.data['train']])
            self.val_X = np.array([datum['thermal_sum'] for datum in self.data['val']])
            self.test_X = np.array([datum['thermal_sum'] for datum in self.data['test']])

        self.train_X_2d = np.array([datum.flatten() for datum in self.train_X])
        self.val_X_2d = np.array([datum.flatten() for datum in self.val_X])
        self.test_X_2d = np.array([datum.flatten() for datum in self.test_X])

        self.train_y = np.array([datum['y'] for datum in self.data['train']])
        self.val_y = np.array([datum['y'] for datum in self.data['val']])
        self.test_y = np.array([datum['y'] for datum in self.data['test']])

        # self.train_svm()
        # self.train_rf()
        # self.train_lr()
        # self.train_sgd()
        # self.train_knn()
        # self.train_dt()
        self.train_mlp()

    # Decision Tree
    def train_dt(self):
        dt = DecisionTreeClassifier()
        dt.fit(self.train_X_2d, self.train_y)
        pred_y = dt.predict(self.val_X_2d)
        print('Decision Tree accuracy:', accuracy_score(self.val_y, pred_y))

    # K Nearest Neighbors
    def train_knn(self):
        knn = KNeighborsClassifier()
        knn.fit(self.train_X_2d, self.train_y)
        pred_y = knn.predict(self.val_X_2d)
        print('KNN accuracy:', accuracy_score(self.val_y, pred_y))

    # Logistic Regression
    def train_lr(self):
        lr = LogisticRegression(max_iter=200)
        lr.fit(self.train_X_2d, self.train_y)
        pred_y = lr.predict(self.val_X_2d)
        print('logistic regression accuracy:', accuracy_score(self.val_y, pred_y))

    # Multi-Layer Perceptron
    def train_mlp(self):
        parameters = {
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam']
            # 'hidden_layer_sizes': []
        }
        mlp = MLPClassifier()
        clf = GridSearchCV(mlp, parameters)
        clf.fit(self.train_X_2d, self.train_y)
        pred_y = clf.predict(self.val_X_2d)
        print('Multi-Layer Perceptron accuracy:', accuracy_score(self.val_y, pred_y))

    # Random Forest
    def train_rf(self):
        parameters = {
            'n_estimators': [50, 100, 250, 500], # 250
            'min_impurity_decrease': [0, 0.25, 0.5], # 0
            'class_weight': [None, 'balanced'] # balanced
        }
        rfc = RandomForestClassifier()
        clf = GridSearchCV(rfc, parameters)
        clf.fit(self.train_X_2d, self.train_y)
        pred_y = clf.predict(self.val_X_2d)
        print('random forest accuracy:', accuracy_score(self.val_y, pred_y))
        print('\tparams:', clf.best_params_)

    # Stochastic Gradient Descent
    def train_sgd(self):
        sgd = SGDClassifier()
        sgd.fit(self.train_X_2d, self.train_y)
        pred_y = sgd.predict(self.val_X_2d)
        print('Stochastic Gradient Descent accuracy:', accuracy_score(self.val_y, pred_y))

    # Support Vector Machine
    def train_svm(self):
        svc = SVC()
        svc.fit(self.train_X_2d, self.train_y)
        pred_y = svc.predict(self.val_X_2d)
        print('svm accuracy:', accuracy_score(self.val_y, pred_y))


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


