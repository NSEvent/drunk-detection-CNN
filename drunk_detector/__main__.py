#!/usr/bin/env python3

# e.g.: python drunk_detector data --train-files data/train/*/* --test-files data/test/*/* --val-files data/validation/*/*
#       python drunk_detector train -d data_2020-03-18_20-59-45.pickle

import argparse
import datetime
import numpy as np
import os
import pickle
from PIL import Image
import re
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Model types
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Tensorflow Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras import optimizers


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
        self.augment_data('train')
        self.augment_data('val')
        self.augment_data('test')
        curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = '{}_{}.pickle'.format('data', curr_datetime)
        with open(os.path.join(self.args.output_dir, filename), 'wb') as out_file:
            pickle.dump(self.data, out_file)

    def augment_data(self, split_set):
        augmented_data = []
        for datum in self.data[split_set]:
            # horizontal flip with a noise on non-zero values
            variance = [[[20 if col else 0 for col in row] for row in layer ] for layer in datum['thermal_frames']]
            thermal_frames = np.random.normal(np.flip(datum['thermal_frames'], 2), variance)
            thermal_sum = np.zeros((128, 160))
            for f, frame in enumerate(thermal_frames):
                min_val = max(np.amin(frame), 0)
                frame -= min_val
                for r, row in enumerate(frame):
                    for c, val in enumerate(row):
                        if val < 0:
                            val = 0
                            thermal_frames[f, r, c] = 0
                        thermal_sum[r, c] += val
            augmented_data.append({
                'filename': datum['filename'] + '_flipped',
                'thermal_sum': thermal_sum,
                'thermal_frames': thermal_frames,
                'y': datum['y']
            })

            # noisy version of image
            variance = [[[100 if col else 0 for col in row] for row in layer ] for layer in datum['thermal_frames']]
            thermal_frames = np.random.normal(datum['thermal_frames'], variance)
            thermal_sum = np.zeros((128, 160))
            for f, frame in enumerate(thermal_frames):
                min_val = max(np.amin(frame), 0)
                frame -= min_val
                for r, row in enumerate(frame):
                    for c, val in enumerate(row):
                        if val < 0:
                            val = 0
                            thermal_frames[f, r, c] = 0
                        thermal_sum[r, c] += val
            augmented_data.append({
                'filename': datum['filename'] + '_noisy',
                'thermal_sum': thermal_sum,
                'thermal_frames': thermal_frames,
                'y': datum['y']
            })

            # blurred version of image
            thermal_frames = gaussian_filter(datum['thermal_frames'], sigma=3)
            thermal_sum = np.zeros((128, 160))
            for _, frame in enumerate(thermal_frames):
                min_val = max(np.amin(frame), 0)
                frame -= min_val
                for r, row in enumerate(frame):
                    for c, val in enumerate(row):
                        thermal_sum[r, c] += val
            augmented_data.append({
                'filename': datum['filename'] + '_blur',
                'thermal_sum': thermal_sum,
                'thermal_frames': thermal_frames,
                'y': datum['y']
            })

            # im = Image.fromarray(thermal_sum/np.max(thermal_sum) * 255)
            # im.show()
            # im_orig = Image.fromarray(datum['thermal_sum']/np.max(datum['thermal_sum']) * 255)
            # im_orig.show()
            # im_orig.show()
            # breakpoint()
        self.data[split_set] += augmented_data

    def read_images(self, files, split_set):
        for filename in files:
            basename = os.path.basename(filename)
            if not self.is_face_image(basename):
                continue

            with Image.open(filename) as img:
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
        # self.train_mlp()
        self.train_cnn()

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

    # Convolutional Neural Network
    def train_cnn(self):
        assert self.args.data_mode == 'sum', \
                'Use [-m sum] for training CNN'

        x, y = self.train_X.shape[1:]

        model = Sequential()
        # Layer 1
        model.add(Conv2D(8, (3, 3), input_shape=(x,y,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Layer 2
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Layer 3
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Output layer
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

        # Treat every sober image with same weight as 3 drunk images
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
        class_weight = {0: (1/4)/2, 1: (3/4)/2}

        #### Below is adapted from class example of CNN
        num_epochs = 1000

        # Holds performance statistics across epochs
        perf_time = np.zeros((num_epochs, 4))

        # Set up figure
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        best_val = [np.inf, 0]
        for epoch in np.arange(0,num_epochs):
            model.fit(cnn_data(self.train_X), np.array(self.train_y),
                      batch_size=128,
                      epochs=1,
                      verbose=1,
                      class_weight=class_weight,
                      validation_data=(cnn_data(self.val_X), np.array(self.val_y)),
                      shuffle=True,
                      use_multiprocessing=True)
            # Check the performance on train/test/val
            # The model.evaluate function returns an array: [loss, accuracy]
            val = model.evaluate(cnn_data(self.val_X), np.array(self.val_y))  # val = [val_loss, val_accuracy]
            new = [model.evaluate(cnn_data(self.train_X), np.array(self.train_y))[1],
                   val[0], val[1],
                   model.evaluate(cnn_data(self.test_X), np.array(self.test_y))[1]]
            perf_time[epoch,:]=new

            # Visualize
            plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,0],'b', label='train')
            plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,2],'r', label='validation')
            plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,3],'g', label='test')
            plt.legend(loc='upper left')
            plt.show()

            # Test if validation performance has improved (val_loss)
            if val[0] >= best_val[0]:
                best_val[1] += 1
            else:
                best_val = [val[0], 0]
            print ("epoch %d, loss %f, number %d" %(epoch, best_val[0], best_val[1]))

            # Stop training if performance hasn't increased in STOP_ITERATIONS
            STOP_ITERATIONS = 30
            if best_val[1] > STOP_ITERATIONS:
                break

        # Export model and test/val/train plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,0],'b', label='train')
        plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,2],'r', label='validation')
        plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,3],'g', label='test')
        plt.legend(loc='upper left')

        curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename_fig = '{}_{}.png'.format('cnn_fig', curr_datetime)
        plt.savefig(filename_fig)
        plt.close('all') # Close fig to save memory

        filename_model = '{}_{}.hdf5'.format('cnn_model', curr_datetime)
        save_model(model, filename_model)

        pred_y = model.predict_classes(cnn_data(self.test_X))
        print('CNN accuracy:', accuracy_score(self.test_y, pred_y))
        print('CNN confusion matrix\n', confusion_matrix(self.test_y, pred_y))

# Only for X data, use np.array(y) for y data
# Returns data formatted for Keras CNN input
def cnn_data(data):
    x, y = data.shape[1:]
    return data.reshape((-1, x, y, 1))

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
        # TODO randomize train and val data here
        dd.train()

    elif args.mode == 'predict':
        # TODO randomize test data here
        pass
