#!/usr/bin/env python3

# e.g.: python drunk_detector data --train-files data/train/*/* --test-files data/test/*/* --val-files data/validation/*/*
#       python drunk_detector train -m sum -d data_2020-04-09_12-20-39.pickle
#       python drunk_detector predict -d data_2020-04-09_12-20-39.pickle -c voter_2020-04-10_20-51-32.pickle
# tensorboard --logdir="logs/"

import argparse
import datetime
import numpy as np
import os
import pickle
from PIL import Image, ImageOps
import re
import itertools
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

from imgaug import augmenters as iaa
from statsmodels.stats.anova import AnovaRM
import pandas as pd

# Model types
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint


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
#         self.augment_data('train')
#         self.augment_data('val')
#         self.augment_data('test')
        # Current data shows that self.augment_data() produces data not suitable for training on CNN
        # Using flip for data augmentation in self.train()
        curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = '{}_{}.pickle'.format('data', curr_datetime)
        with open(os.path.join(self.args.output_dir, filename), 'wb') as out_file:
            pickle.dump(self.data, out_file)

    def augment_data(self, split_set):

        augmented_data = []
        for datum in self.data[split_set]:
            for _ in range(1 if datum['y'] else 1):
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
                variance = [[[25 if col else 0 for col in row] for row in layer ] for layer in datum['thermal_frames']]
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
        if re.match(self.sober_filename_patter, filename) is not None:
            return 0
        else:
            return 1

    def predict(self):
        # pass
        # with open(self.args.classifier, 'rb') as clf_file:
        #     self.clf = pickle.load(clf_file)
        #
        # with open(self.args.data, 'rb') as data_file:
        #     self.data = pickle.load(data_file)
        #
        # if self.args.data_mode == 'frames':
        #     self.test_X = np.array([datum['thermal_frames'] for datum in self.data['test']])
        # elif self.args.data_mode == 'sum':
        #     self.test_X = np.array([datum['thermal_sum'] for datum in self.data['test']])
        #
        # self.test_X_2d = np.array([datum.flatten() for datum in self.test_X])
        # self.test_y = np.array([datum['y'] for datum in self.data['test']])
        #
        # pred_y =  self.clf.predict(self.test_X_2d)
        # print('Classifier accuracy:', accuracy_score(self.test_y, pred_y))
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

        # Flip data
        flip_seq = iaa.Sequential([
            iaa.Fliplr(1), # horizontally flip all of the images
        ])
        flip_data = flip_seq(images=self.train_X)
        self.train_X = np.concatenate((self.train_X, flip_data), axis=0)
        self.train_y = np.concatenate((self.train_y, self.train_y), axis=0)
        flip_data = flip_seq(images=self.test_X)
        self.test_X = np.concatenate((self.test_X, flip_data), axis=0)
        self.test_y = np.concatenate((self.test_y, self.test_y), axis=0)
        flip_data = flip_seq(images=self.val_X)
        self.val_X = np.concatenate((self.val_X, flip_data), axis=0)
        self.val_y = np.concatenate((self.val_y, self.val_y), axis=0)

        # Shuffle data
        self.train_X, self.train_y = shuffle_pair(self.train_X, self.train_y)
        self.val_X, self.val_y = shuffle_pair(self.val_X, self.val_y)
        self.test_X, self.test_y = shuffle_pair(self.test_X, self.test_y)

        self.test_best_model()

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

        # Flip data
        flip_seq = iaa.Sequential([
            iaa.Fliplr(1), # horizontally flip all of the images
        ])
        flip_data = flip_seq(images=self.train_X)
        self.train_X = np.concatenate((self.train_X, flip_data), axis=0)
        self.train_y = np.concatenate((self.train_y, self.train_y), axis=0)
        flip_data = flip_seq(images=self.test_X)
        self.test_X = np.concatenate((self.test_X, flip_data), axis=0)
        self.test_y = np.concatenate((self.test_y, self.test_y), axis=0)
        flip_data = flip_seq(images=self.val_X)
        self.val_X = np.concatenate((self.val_X, flip_data), axis=0)
        self.val_y = np.concatenate((self.val_y, self.val_y), axis=0)

        # Shuffle data
        self.train_X, self.train_y = shuffle_pair(self.train_X, self.train_y)
        self.val_X, self.val_y = shuffle_pair(self.val_X, self.val_y)
        self.test_X, self.test_y = shuffle_pair(self.test_X, self.test_y)

        # svc = self.train_svm()
        # rfc = self.train_rf()
        # lr = self.train_lr()
        # sgd = self.train_sgd()
        # knn = self.train_knn()
        # dt = self.train_dt()
        # voter = self.train_voter()
        # mlp = self.train_mlp()
        cnn = self.train_cnn_hyperparameters()

        # curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # filename = '{}_{}.pickle'.format(self.args.name, curr_datetime)
        # with open(os.path.join(self.args.output_dir, filename), 'wb') as out_file:
        #     pickle.dump(voter, out_file)


    # Voting Classifier
    def train_voter(self):
        dt = DecisionTreeClassifier()
        knn = KNeighborsClassifier()
        lr = LogisticRegression(max_iter=200)
        rfc = RandomForestClassifier(class_weight='balanced', n_estimators=250, min_impurity_decrease=0)
        sgd = SGDClassifier(loss='log')
        svc = SVC(probability=True)
        voter = VotingClassifier(
            estimators=[('dt', dt), ('knn', knn), ('lr', lr), ('rfc', rfc), ('sgd', sgd), ('svc', svc)],
            voting='soft')

        voter.fit(self.train_X_2d, self.train_y)
        pred_y = voter.predict(self.val_X_2d)
        # print('voter accuracy:', accuracy_score(self.val_y, pred_y))
        return voter

    # Decision Tree
    def train_dt(self):
        dt = DecisionTreeClassifier()
        dt.fit(self.train_X_2d, self.train_y)
        # pred_y = dt.predict(self.val_X_2d)
        # print('Decision Tree accuracy:', accuracy_score(self.val_y, pred_y))
        return dt

    # K Nearest Neighbors
    def train_knn(self):
        knn = KNeighborsClassifier()
        knn.fit(self.train_X_2d, self.train_y)
        # pred_y = knn.predict(self.val_X_2d)
        # print('KNN accuracy:', accuracy_score(self.val_y, pred_y))
        return knn

    # Logistic Regression
    def train_lr(self):
        lr = LogisticRegression(max_iter=200)
        lr.fit(self.train_X_2d, self.train_y)
        # pred_y = lr.predict(self.val_X_2d)
        # print('logistic regression accuracy:', accuracy_score(self.val_y, pred_y))
        return lr

    # Multi-Layer Perceptron
    def train_mlp(self):
        parameters = {
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam']
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
        rfc = RandomForestClassifier(class_weight='balanced', n_estimators=250, min_impurity_decrease=0)
        # clf = GridSearchCV(rfc, parameters)
        rfc.fit(self.train_X_2d, self.train_y)
        # pred_y = rfc.predict(self.val_X_2d)
        # print('random forest accuracy:', accuracy_score(self.val_y, pred_y))
        # print('\tparams:', clf.best_params_)
        return rfc

    # Stochastic Gradient Descent
    def train_sgd(self):
        sgd = SGDClassifier()
        sgd.fit(self.train_X_2d, self.train_y)
        # pred_y = sgd.predict(self.val_X_2d)
        # print('Stochastic Gradient Descent accuracy:', accuracy_score(self.val_y, pred_y))
        return sgd

    # Support Vector Machine
    def train_svm(self):
        svc = SVC()
        svc.fit(self.train_X_2d, self.train_y)

        # pred_y = svc.predict(self.val_X_2d)
        # print('svm accuracy:', accuracy_score(self.val_y, pred_y))
        return svc


    def prepare_data_cnn(self):
        assert self.args.data_mode == 'sum', \
                'Use [-m sum] for training CNN'

        self.train_X = tf.keras.utils.normalize(self.train_X, axis=1)
        self.val_X = tf.keras.utils.normalize(self.val_X, axis=1)
        self.test_X = tf.keras.utils.normalize(self.test_X, axis=1)

    # Convolutional Neural Network
    def train_cnn(self):
        self.prepare_data_cnn()

        curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        x, y = self.train_X.shape[1:]

        model = Sequential()
        # Layer 1
        model.add(Conv2D(8, (3, 3), input_shape=(x,y,1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Layer 2
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Layer 3
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Dense layer
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())

        # Output layer
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                        optimizer=optimizers.Adam(learning_rate=0.001),
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
                      batch_size=32,
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
            # Model improved
            else:
                best_val = [val[0], 0]
                # Save current best model
                filename_model = '{}_{}_epoch={}_val_acc={}.hdf5'.format('cnn_model', curr_datetime, epoch, val[1])
                save_model(model, filename_model)
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

        filename_fig = '{}_{}.png'.format('cnn_fig', curr_datetime)
        plt.savefig(filename_fig)
        plt.close('all') # Close fig to save memory

        filename_model = '{}_{}.hdf5'.format('cnn_model', curr_datetime)
        save_model(model, filename_model)

        pred_y = model.predict_classes(cnn_data(self.test_X))
        print('CNN accuracy:', accuracy_score(self.test_y, pred_y))
        print('CNN confusion matrix\n', confusion_matrix(self.test_y, pred_y))



    # Determine if a prediction is statistically significantly better than predicting all drunk
    def test_model_significance(self, pred_y):
        subject = [] # Keep track of person
        correct = [] # Keep track of images correctly predicted
        model_name = [] # Keep track of corresponding model name

        for i, (true, pred) in enumerate(zip(self.test_y, pred_y)):
            subject.append(i)
            correct.append(int(true==pred))
            model_name.append('best')
        pred_drunk = []
        for _ in range(len(self.test_y)):
            pred_drunk.append(1)
        for i, (true, pred) in enumerate(zip(self.test_y, pred_drunk)):
            subject.append(i)
            correct.append(int(true==pred))
            model_name.append('drunk')

        anova_dict = {'Correct/Incorrect':correct,'Test_ID':subject,'Model_Name':model_name}
        anova_df = pd.DataFrame(anova_dict)

        anovarm = AnovaRM(data=anova_df, depvar='Correct/Incorrect', subject='Test_ID', within=['Model_Name'])
        fit = anovarm.fit()
        print(fit.summary())


    def demo_model(self, pred_y):
        print('Making predictions on 70 images in test set')

        state = ['sober', 'drunk']
        for i,(actual,pred) in enumerate(zip(self.test_y ,pred_y)):
            if i > 0:
                print(); print()

            bw = Image.fromarray(self.test_X[i]*1500).convert('L')
            add_border
            plt.imshow(add_border(bw, 25, int(actual)==int(pred)))
            print(f'Actual: {state[int(actual)]}, Predicted: {state[int(pred)]}', flush=True)
            input('Enter for next prediction:')


    def test_best_model(self):
        self.prepare_data_cnn()

        model = load_model('models/n_clayers=1,clayer_sz=8,n_dlayers=0,dlayer_sz=512,lr=0.01,bat_size=32FINAL-test_acc=0.8714285492897034.hdf5')
        pred_y = model.predict_classes(cnn_data(self.test_X))
        print('CNN accuracy:', accuracy_score(self.test_y, pred_y))
        print('CNN confusion matrix\n', confusion_matrix(self.test_y, pred_y))
        self.test_model_significance(pred_y)

        plot_confusion_matrix(confusion_matrix(self.test_y, pred_y), target_names=['sober', 'drunk'])
        import time; time.sleep(1)
        self.demo_model(pred_y)


    # Convolutional Neural Network hyperparameter tuning
    def train_cnn_hyperparameters(self):
        self.prepare_data_cnn()

        curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        x, y = self.train_X.shape[1:]


        # saved_acc = pickle.load(open('saved_acc.pickle, 'rb'))

        # hyperparameters
        num_convlayers = [1, 2, 3]
        convlayer_size = [8, 16, 32]
        num_denselayers = [0, 1, 2]
        denselayer_size = [128, 256, 512]
        learning_rate = [0.01, 0.001, 0.0001]
        batch_size = [16, 32]

        # creates a list of all combinations of hyperparameters
        param_grid = list(itertools.product(
            num_convlayers, convlayer_size,
            num_denselayers, denselayer_size,
            learning_rate,
            batch_size))

        print(f'CNN hyperparameter tuning with {len(param_grid)} combinations')

        # offset = len(saved_acc)
        for i, params in enumerate(shuffle(param_grid)):

            print(f'Run {i}/{len(param_grid)}')

            # tuning parameters
            num_convlayers, \
            convlayer_size, \
            num_denselayers, \
            denselayer_size, \
            learning_rate, \
            batch_size = params

            NAME = f'n_clayers={num_convlayers},clayer_sz={convlayer_size},n_dlayers={num_denselayers},dlayer_sz={denselayer_size},lr={learning_rate},bat_size={batch_size}'
            tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

#             save_best_model_loss = ModelCheckpoint(NAME + "val_loss={val_loss:.4f}.hdf5",
#                                                   monitor='val_loss',
#                                                   verbose=0, save_best_only=True,
#                                                   save_weights_only=False,
#                                                   mode='auto', save_freq='epoch')
#             save_best_model_acc = tf.keras.callbacks.ModelCheckpoint(NAME + "val_acc={val_accuracy:.4f}.hdf5",
#                                                      monitor='val_accuracy',
#                                                      verbose=0, save_best_only=True,
#                                                      save_weights_only=False,
#                                                      mode='auto', save_freq='epoch')
            early_stopping = tf.keras.callbacks.EarlyStopping(
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    patience=30,
                                                    mode='auto',
                                                    restore_best_weights=False)

            model = Sequential()
            # Conv layer 1
            model.add(Conv2D(convlayer_size, (3, 3), input_shape=(x,y,1)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Additional Conv layers
            for _ in range(num_convlayers-1):
                model.add(Conv2D(convlayer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # Dense layer 1
            model.add(Flatten())
            for _ in range(num_denselayers):
                model.add(Dense(denselayer_size, activation='relu'))

            if False:
                model.add(BatchNormalization())

            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            model.compile(loss='binary_crossentropy',
                            optimizer=optimizers.Adam(learning_rate=learning_rate),
                            metrics=['accuracy'])

            # Treat every sober image with same weight as 3 drunk images
            # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights
            class_weight = {0: (1/4)/2, 1: (3/4)/2}

            num_epochs = 1000
            model.fit(cnn_data(self.train_X), np.array(self.train_y),
                      batch_size=batch_size,
                      epochs=num_epochs,
                      verbose=1,
                      class_weight=class_weight,
                      validation_data=(cnn_data(self.val_X), np.array(self.val_y)),
                      shuffle=True,
                      use_multiprocessing=True,
                      callbacks=[tensorboard, early_stopping])

            save_model(model, NAME + f'FINAL-test_acc={model.evaluate(cnn_data(self.test_X),self.test_y)[1]}.hdf5')


# For use in demo to border images
def add_border(pil_image, border, match):
    color = ''
    if match:
        color = 'green'
    else:
        color = 'red'

    pil_image = pil_image.convert('RGB')

    if isinstance(border, int) or isinstance(border, tuple):
        bimg = ImageOps.expand(pil_image, border=border, fill=color)
    else:
        raise RuntimeError('Border is not an integer or tuple!')
    return bimg


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_matrix.png')


# Shuffles a pair of equal size arrays in the same random order
# Returns a tuple of shuffled arrays
def shuffle_pair(X_arr, y_arr):
    X_arr_s = []
    y_arr_s = []
    for X, y in shuffle(list(zip(X_arr, y_arr))):
        X_arr_s.append(X)
        y_arr_s.append(y)
    return (np.array(X_arr_s), np.array(y_arr_s))


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
    parser.add_argument('-n', '--name', type=str, default='clf', help='name of trained classifier')
    parser.add_argument('-c', '--classifier', type=str, help='trained classifier for predictions')
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
        dd.predict();
        pass
