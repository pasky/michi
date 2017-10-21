from __future__ import print_function

import numpy as np
import random
import time

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.merge import add


def flip_vert(board):
    return '\n'.join(reversed(board[:-1].split('\n'))) + ' '


def flip_horiz(board):
    return '\n'.join([' ' + l[1:][::-1] for l in board.split('\n')])


def flip_both(board):
    return '\n'.join(reversed([' ' + l[1:][::-1] for l in board[:-1].split('\n')])) + ' '


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
    def __init__(self, N, batch_size=32):
        self.N = N
        self.batch_size = 32
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

    def fit_game(self, positions, result, board_transform=None):
        X, y_dist, y_res = [], [], []
        for pos, dist in random.sample(positions, len(positions)):
            X.append(self._X_position(pos, board_transform=board_transform))
            y_dist.append(dist)
            y_res.append(float(result) / 2 + 0.5)
            if len(X) % self.batch_size == 0:
                self.model.train_on_batch(np.array(X), [np.array(y_dist), np.array(y_res)])
                X, y_dist, y_res = [], [], []
            result = -result
        if len(X) > 0:
            self.model.train_on_batch(np.array(X), [np.array(y_dist), np.array(y_res)])

    def predict(self, position):
        X = self._X_position(position)
        return self.model.predict(np.array([X]))

    def predict_distribution(self, position):
        return self.predict(position)[0][0]

    def predict_winrate(self, position):
        return self.predict(position)[1][0][0] * 2 - 1

    def _X_position(self, position, board_transform=None):
        N = self.N
        W = N + 2
        my_stones, their_stones, edge, last, last2, to_play = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
        board = position.board
        if board_transform:
            board = eval(board_transform)(board)
        for c, p in enumerate(board):
            x, y = c % W - 1, c // W - 1
            # In either case, y and x should be sane (not off-board)
            if p == 'X':
                my_stones[y, x] = 1
            elif p == 'x':
                their_stones[y, x] = 1
            if not (x >= 0 and x < N and y >= 0 and y < N):
                continue
            if x == 0 or x == N-1 or y == 0 or y == N-1:
                edge[y, x] = 1
            if position.last == c:
                last[y, x] = 1
            if position.last2 == c:
                last2[y, x] = 1
            if position.n % 2 == 1:
                to_play[y, x] = 1
        return np.stack((my_stones, their_stones, edge, last, last2, to_play), axis=-1)
