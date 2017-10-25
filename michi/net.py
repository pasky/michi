from __future__ import print_function

import itertools
import joblib
import numpy as np
import random
import time

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.merge import add


############################
# AlphaGo Zero style network

class ResNet(object):
    def __init__(self, input_N=256, filter_N=256, n_stages=19,
                 kernel_width=3, kernel_height=3,
                 inpkern_width=3, inpkern_height=3):
        # number of filters and dimensions of the initial input kernel
        self.input_N = input_N
        self.inpkern_width = inpkern_width
        self.inpkern_height = inpkern_height
        # base number of filters and dimensions of the followup resnet kernels
        self.filter_N = filter_N
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.n_stages = n_stages

    def create(self, width, height, n_planes):
        bn_axis = 3
        inp = Input(shape=(width, height, n_planes))

        x = inp
        x = Conv2D(self.input_N, (self.inpkern_width, self.inpkern_height), padding='same', name='conv1')(x)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)

        for i in range(self.n_stages):
            x = self.identity_block(x, [self.filter_N, self.filter_N], stage=i+1, block='a')

        self.model = Model(inp, x)
        return self.model

    def identity_block(self, input_tensor, filters, stage, block):
        '''The identity_block is the block that has no conv layer at shortcut

        # Arguments
            input_tensor: input tensor
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        '''
        nb_filter1, nb_filter2 = filters
        bn_axis = 3
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = input_tensor
        x = Conv2D(nb_filter1, (self.kernel_width, self.kernel_height),
                          padding='same', name=conv_name_base + 'a')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (self.kernel_width, self.kernel_height),
                          padding='same', name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)
        return x


class AGZeroModel:
    def __init__(self, N, batch_size=32, archive_fit_samples=48):
        self.N = N
        self.batch_size = 32

        self.archive_fit_samples = archive_fit_samples
        self.position_archive = []

        self.model_name = time.strftime('G%y%m%dT%H%M%S')
        print(self.model_name)

    def create(self):
        bn_axis = 3

        N = self.N
        position = Input((N, N, 6))
        resnet = ResNet(n_stages=N)
        resnet.create(N, N, 6)
        x = resnet.model(position)

        dist = Conv2D(2, (1, 1))(x)
        dist = BatchNormalization(axis=bn_axis)(dist)
        dist = Activation('relu')(dist)
        dist = Flatten()(dist)
        dist = Dense(N * N, activation='softmax')(dist)
        dist = Reshape((N, N), name='distribution')(dist)

        res = Conv2D(1, (1, 1))(x)
        res = BatchNormalization(axis=bn_axis)(res)
        res = Activation('relu')(res)
        res = Flatten()(res)
        res = Dense(256, activation='relu')(res)
        res = Dense(1, activation='sigmoid', name='result')(res)

        self.model = Model(position, [dist, res])
        self.model.compile('adam', ['categorical_crossentropy', 'binary_crossentropy'])
        self.model.summary()

    def fit_game(self, X_positions, result):
        self.position_archive.extend(X_positions)

        if len(self.position_archive) >= self.archive_fit_samples:
            archive_samples = random.sample(self.position_archive, self.archive_fit_samples)
        else:
            # initial case
            archive_samples = self.position_archive
        # I'm going to some lengths to avoid the potentially overloaded + operator
        X_fit_samples = list(itertools.chain(X_positions, archive_samples))
        X_shuffled = random.sample(X_fit_samples, len(X_fit_samples))

        X, y_dist, y_res = [], [], []
        for pos, dist in X_shuffled:
            X.append(pos)
            y_dist.append(dist)
            y_res.append(float(result) / 2 + 0.5)
            if len(X) % self.batch_size == 0:
                self.model.train_on_batch(np.array(X), [np.array(y_dist), np.array(y_res)])
                X, y_dist, y_res = [], [], []
            result = -result
        if len(X) > 0:
            self.model.train_on_batch(np.array(X), [np.array(y_dist), np.array(y_res)])

    def predict(self, X_positions):
        dist, res = self.model.predict(X_positions)
        res = np.array([r[0] * 2 - 1 for r in res])
        return [dist, res]

    def save(self, snapshot_id):
        self.model.save_weights('%s.weights.h5' % (snapshot_id,))
        joblib.dump(self.position_archive, '%s.archive.joblib' % (snapshot_id,), compress=5)

    def load(self, snapshot_id):
        self.model.load_weights('%s.weights.h5' % (snapshot_id,))

        pos_fname = '%s.archive.joblib' % (snapshot_id,)
        try:
            self.position_archive = joblib.load(pos_fname)
        except:
            print('Warning: Cannot load position archive %s' % (pos_fname,))
