#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
#
# (c) Petr Baudis <pasky@ucw.cz>  2015
# MIT licence (i.e. almost public domain)
#
# A minimalistic Go-playing engine attempting to strike a balance between
# brevity, educational value and strength.  It can beat GNUGo on 13x13 board
# on a modest 4-thread laptop.
#
# When benchmarking, note that at the beginning of the first move the program
# runs much slower because pypy is JIT compiling on the background!
#
# To start reading the code, begin either:
# * Bottom up, by looking at the goban implementation - starting with
#   the 'empty' definition below and Position.move() method.
# * In the middle, by looking at the Monte Carlo playout implementation,
#   starting with the mcplayout() function.
# * Top down, by looking at the MCTS implementation, starting with the
#   tree_search() function.  It can look a little confusing due to the
#   parallelization, but really is just a loop of tree_descend(),
#   mcplayout() and tree_update() round and round.
# It may be better to jump around a bit instead of just reading straight
# from start to end.

from __future__ import print_function
from collections import namedtuple
from itertools import count
import math
import multiprocessing
from multiprocessing.pool import Pool
import random
import re
import sys
import time

from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dense, Flatten, Input, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.merge import add
import numpy as np


# Given a board of size NxN (N=9, 19, ...), we represent the position
# as an (N+1)*(N+2) string, with '.' (empty), 'X' (to-play player),
# 'x' (other player), and whitespace (off-board border to make rules
# implementation easier).  Coordinates are just indices in this string.
# You can simply print(board) when debugging.
N = 5
W = N + 2
empty = "\n".join([(N+1)*' '] + N*[' '+N*'.'] + [(N+2)*' '])
colstr = 'ABCDEFGHJKLMNOPQRST'

N_SIMS = 200
PUCT_C = 0.1
PROPORTIONAL_STAGE = 5
RAVE_EQUIV = 100
EXPAND_VISITS = 1
PRIOR_EVEN = 4  # should be even number; 0.5 prior
PRIOR_NET = 40
REPORT_PERIOD = 200
RESIGN_THRES = 0.2


#######################
# board string routines

def neighbors(c):
    """ generator of coordinates for all neighbors of c """
    return [c-1, c+1, c-W, c+W]

def diag_neighbors(c):
    """ generator of coordinates for all diagonal neighbors of c """
    return [c-W-1, c-W+1, c+W-1, c+W+1]


def board_put(board, c, p):
    return board[:c] + p + board[c+1:]


def floodfill(board, c):
    """ replace continuous-color area starting at c with special color # """
    # This is called so much that a bytearray is worthwhile...
    byteboard = bytearray(board)
    p = byteboard[c]
    byteboard[c] = ord('#')
    fringe = [c]
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if byteboard[d] == p:
                byteboard[d] = ord('#')
                fringe.append(d)
    return str(byteboard)


# Regex that matches various kind of points adjecent to '#' (floodfilled) points
contact_res = dict()
for p in ['.', 'x', 'X']:
    rp = '\\.' if p == '.' else p
    contact_res_src = ['#' + rp,  # p at right
                       rp + '#',  # p at left
                       '#' + '.'*(W-1) + rp,  # p below
                       rp + '.'*(W-1) + '#']  # p above
    contact_res[p] = re.compile('|'.join(contact_res_src), flags=re.DOTALL)

def contact(board, p):
    """ test if point of color p is adjecent to color # anywhere
    on the board; use in conjunction with floodfill for reachability """
    m = contact_res[p].search(board)
    if not m:
        return None
    return m.start() if m.group(0)[0] == p else m.end() - 1


def is_eyeish(board, c):
    """ test if c is inside a single-color diamond and return the diamond
    color or None; this could be an eye, but also a false one """
    eyecolor = None
    for d in neighbors(c):
        if board[d].isspace():
            continue
        if board[d] == '.':
            return None
        if eyecolor is None:
            eyecolor = board[d]
            othercolor = eyecolor.swapcase()
        elif board[d] == othercolor:
            return None
    return eyecolor

def is_eye(board, c):
    """ test if c is an eye and return its color or None """
    eyecolor = is_eyeish(board, c)
    if eyecolor is None:
        return None

    # Eye-like shape, but it could be a falsified eye
    falsecolor = eyecolor.swapcase()
    false_count = 0
    at_edge = False
    for d in diag_neighbors(c):
        if board[d].isspace():
            at_edge = True
        elif board[d] == falsecolor:
            false_count += 1
    if at_edge:
        false_count += 1
    if false_count >= 2:
        return None

    return eyecolor


class Position(namedtuple('Position', 'board cap n ko last last2 komi')):
    """ Implementation of simple Chinese Go rules;
    n is how many moves were played so far """

    def move(self, c):
        """ play as player X at the given coord c, return the new position """

        # Test for ko
        if c == self.ko:
            return None
        # Are we trying to play in enemy's eye?
        in_enemy_eye = is_eyeish(self.board, c) == 'x'

        board = board_put(self.board, c, 'X')
        # Test for captures, and track ko
        capX = self.cap[0]
        singlecaps = []
        for d in neighbors(c):
            if board[d] != 'x':
                continue
            # XXX: The following is an extremely naive and SLOW approach
            # at things - to do it properly, we should maintain some per-group
            # data structures tracking liberties.
            fboard = floodfill(board, d)  # get a board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            # no liberties left for this group, remove the stones!
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Set ko
        ko = singlecaps[0] if in_enemy_eye and len(singlecaps) == 1 else None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        # Update the position and return
        return Position(board=board.swapcase(), cap=(self.cap[1], capX),
                        n=self.n + 1, ko=ko, last=c, last2=self.last, komi=self.komi)

    def pass_move(self):
        """ pass - i.e. return simply a flipped position """
        return Position(board=self.board.swapcase(), cap=(self.cap[1], self.cap[0]),
                        n=self.n + 1, ko=None, last=None, last2=self.last, komi=self.komi)

    def moves(self, i0):
        """ Generate a list of moves (includes false positives - suicide moves;
        does not include true-eye-filling moves), starting from a given board
        index (that can be used for randomization) """
        i = i0-1
        passes = 0
        while True:
            i = self.board.find('.', i+1)
            if passes > 0 and (i == -1 or i >= i0):
                break  # we have looked through the whole board
            elif i == -1:
                i = 0
                passes += 1
                continue  # go back and start from the beginning
            # Test for to-play player's one-point eye
            if is_eye(self.board, i) == 'X':
                continue
            yield i

    def last_moves_neighbors(self):
        """ generate a randomly shuffled list of points including and
        surrounding the last two moves (but with the last move having
        priority) """
        clist = []
        for c in self.last, self.last2:
            if c is None:  continue
            dlist = [c] + list(neighbors(c) + diag_neighbors(c))
            random.shuffle(dlist)
            clist += [d for d in dlist if d not in clist]
        return clist

    def score(self, owner_map=None):
        """ compute score for to-play player; this assumes a final position
        with all dead stones captured; if owner_map is passed, it is assumed
        to be an array of statistics with average owner at the end of the game
        (+1 black, -1 white) """
        board = self.board
        i = 0
        while True:
            i = self.board.find('.', i+1)
            if i == -1:
                break
            fboard = floodfill(board, i)
            # fboard is board with some continuous area of empty space replaced by #
            touches_X = contact(fboard, 'X') is not None
            touches_x = contact(fboard, 'x') is not None
            if touches_X and not touches_x:
                board = fboard.replace('#', 'X')
            elif touches_x and not touches_X:
                board = fboard.replace('#', 'x')
            else:
                board = fboard.replace('#', ':')  # seki, rare
            # now that area is replaced either by X, x or :
        komi = self.komi if self.n % 2 == 1 else -self.komi
        if owner_map is not None:
            for c in range(W*W):
                n = 1 if board[c] == 'X' else -1 if board[c] == 'x' else 0
                owner_map[c] += n * (1 if self.n % 2 == 0 else -1)
        return board.count('X') - board.count('x') + komi


def empty_position():
    """ Return an initial board position """
    return Position(board=empty, cap=(0, 0), n=0, ko=None, last=None, last2=None, komi=0.5)  #7.5)


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
    def __init__(self, batch_size=32):
        self.batch_size = 32
        self.model_name = time.strftime('G%y%m%dT%H%M%S')
        print(self.model_name)

    def create(self):
        bn_axis = 3

        position = Input((N, N, 3))
        resnet = ResNet(n_stages=N)
        resnet.create(N, N, 3)
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
        for pos, dist in positions:
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
        my_stones, their_stones, edge = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
        board = position.board
        if board_transform:
            board = board_transform(board)
        for c, p in enumerate(board):
            x, y = c % W - 1, c // W - 1
            # In either case, y and x should be sane (not off-board)
            if p == 'X':
                my_stones[y, x] = 1
            elif p == 'x':
                their_stones[y, x] = 1
            if (x >= 0 and x < N and y >= 0 and y < N) and (x == 0 or x == N-1 or y == 0 or y == N-1):
                edge[y, x] = 1
        return np.stack((my_stones, their_stones, edge), axis=-1)


########################
# montecarlo tree search

class TreeNode():
    """ Monte-Carlo tree node;
    v is #visits, w is #wins for to-play (expected reward is w/v)
    pv, pw are prior values (node value = w/v + pw/pv)
    av, aw are amaf values ("all moves as first", used for the RAVE tree policy)
    children is None for leaf nodes """
    def __init__(self, pos):
        self.pos = pos
        self.v = 0
        self.w = 0
        self.pv = 0
        self.pw = 0
        self.av = 0
        self.aw = 0
        self.children = None

    def expand(self):
        """ add and initialize children to a leaf node """
        distribution = net.predict_distribution(self.pos)
        self.children = []
        for c in self.pos.moves(0):
            pos2 = self.pos.move(c)
            if pos2 is None:
                continue
            node = TreeNode(pos2)
            self.children.append(node)

            x, y = c % W - 1, c // W - 1
            value = distribution[y, x]

            node.pv = PRIOR_NET
            node.pw = PRIOR_NET * value

        if not self.children:
            # No possible moves, add a pass move
            self.children.append(TreeNode(self.pos.pass_move()))

    def puct_urgency(self, n0):
        expectation = float(self.w + PRIOR_EVEN/2) / (self.v + PRIOR_EVEN)
        try:
            prior = float(self.pw) / self.pv
        except:
            prior = 0.1  # XXX
        return expectation + PUCT_C * prior * math.sqrt(n0) / (1 + self.v)

    def rave_urgency(self):
        v = self.v + self.pv
        expectation = float(self.w+self.pw) / v
        if self.av == 0:
            return expectation
        rave_expectation = float(self.aw) / self.av
        beta = self.av / (self.av + v + float(v) * self.av / RAVE_EQUIV)
        return beta * rave_expectation + (1-beta) * expectation

    def winrate(self):
        return float(self.w) / self.v if self.v > 0 else float('nan')

    def prior(self):
        return float(self.pw) / self.pv if self.pv > 0 else float('nan')

    def best_move(self, proportional=False):
        """ best move is the most simulated one """
        if self.children is None:
            return None
        if proportional:
            probs = [float(node.v) / self.v for node in self.children]
            probs_tot = sum(probs)
            probs = [p / probs_tot for p in probs]
            i = np.random.choice(len(self.children), p=probs)
            return self.children[i]
        else:
            return max(self.children, key=lambda node: node.v)


def tree_descend(tree, amaf_map, disp=False):
    """ Descend through the tree to a leaf """
    tree.v += 1
    nodes = [tree]
    passes = 0
    while nodes[-1].children is not None and passes < 2:
        if disp:  print_pos(nodes[-1].pos)

        # Pick the most urgent child
        children = list(nodes[-1].children)
        if disp:
            for c in children:
                dump_subtree(c, recurse=False)
        random.shuffle(children)  # randomize the max in case of equal urgency
        dirichlet = np.random.dirichlet((0.03,1), len(children))
        urgencies = [node.puct_urgency(nodes[-1].v)*0.75 + 0.25*dir[0] for node, dir in zip(children, dirichlet)]
        node = max(zip(children, urgencies), key=lambda t: t[1])[0]
        nodes.append(node)

        if disp:  print('chosen %s' % (str_coord(node.pos.last),), file=sys.stderr)
        if node.pos.last is None:
            passes += 1
        else:
            passes = 0
            if amaf_map[node.pos.last] == 0:  # Mark the coordinate with 1 for black
                amaf_map[node.pos.last] = 1 if nodes[-2].pos.n % 2 == 0 else -1

        # updating visits on the way *down* represents "virtual loss", relevant for parallelization
        node.v += 1
        if node.children is None and node.v > EXPAND_VISITS:
            node.expand()

    return nodes


def tree_update(nodes, amaf_map, score, disp=False):
    """ Store simulation result in the tree (@nodes is the tree path) """
    for node in reversed(nodes):
        if disp:  print('updating', str_coord(node.pos.last), score < 0, file=sys.stderr)
        node.w += score < 0  # score is for to-play, node statistics for just-played
        # Update the node children AMAF stats with moves we made
        # with their color
        amaf_map_value = 1 if node.pos.n % 2 == 0 else -1
        if node.children is not None:
            for child in node.children:
                if child.pos.last is None:
                    continue
                if amaf_map[child.pos.last] == amaf_map_value:
                    if disp:  print('  AMAF updating', str_coord(child.pos.last), score > 0, file=sys.stderr)
                    child.aw += score > 0  # reversed perspective
                    child.av += 1
        score = -score


def tree_search(tree, n, owner_map, disp=False):
    """ Perform MCTS search from a given position for a given #iterations """
    # Initialize root node
    if tree.children is None:
        tree.expand()

    i = 0
    while i < n:
        amaf_map = W*W*[0]
        nodes = tree_descend(tree, amaf_map, disp=disp)

        i += 1
        if i > 0 and i % REPORT_PERIOD == 0:
            print_tree_summary(tree, i, f=sys.stderr)

        last_node = nodes[-1]
        if last_node.pos.last is None and last_node.pos.last2 is None:
            score = 1 if last_node.pos.score() > 0 else -1
            #game_score = score
            #if last_node.pos.n % 2:
            #    game_score = -game_score
            #print_pos(last_node.pos, sys.stdout, owner_map)
            #print('[TREE] Two passes, score: B%+.1f (%d)' % (game_score, score))
            #count = last_node.pos.score()
            #if last_node.pos.n % 2:
            #    count = -count
            #print('[TREE] Counted score: B%+.1f' % (count,))
        else:
            score = net.predict_winrate(last_node.pos)
            #print('[TREE] Predicted score: B%+.4f' % (score,))
            #count = last_node.pos.score()
            #if last_node.pos.n % 2:
            #    count = -count
            #print('[TREE] Counted score: B%+.1f' % (count,))

        tree_update(nodes, amaf_map, score, disp=disp)

    if disp:
        dump_subtree(tree)
    if i % REPORT_PERIOD != 0:
        print_tree_summary(tree, i, f=sys.stderr)
    return tree.best_move(tree.pos.n <= PROPORTIONAL_STAGE)


###################
# user interface(s)

# utility routines

def print_pos(pos, f=sys.stderr, owner_map=None):
    """ print visualization of the given board position, optionally also
    including an owner map statistic (probability of that area of board
    eventually becoming black/white) """
    if pos.n % 2 == 0:  # to-play is black
        board = pos.board.replace('x', 'O')
        Xcap, Ocap = pos.cap
    else:  # to-play is white
        board = pos.board.replace('X', 'O').replace('x', 'X')
        Ocap, Xcap = pos.cap
    print('Move: %-3d   Black: %d caps   White: %d caps  Komi: %.1f' % (pos.n, Xcap, Ocap, pos.komi), file=f)
    pretty_board = ' '.join(board.rstrip()) + ' '
    if pos.last is not None:
        pretty_board = pretty_board[:pos.last*2-1] + '(' + board[pos.last] + ')' + pretty_board[pos.last*2+2:]
    rowcounter = count()
    pretty_board = [' %-02d%s' % (N-i, row[2:]) for row, i in zip(pretty_board.split("\n")[1:], rowcounter)]
    if owner_map is not None:
        pretty_ownermap = ''
        for c in range(W*W):
            if board[c].isspace():
                pretty_ownermap += board[c]
            elif owner_map[c] > 0.6:
                pretty_ownermap += 'X'
            elif owner_map[c] > 0.3:
                pretty_ownermap += 'x'
            elif owner_map[c] < -0.6:
                pretty_ownermap += 'O'
            elif owner_map[c] < -0.3:
                pretty_ownermap += 'o'
            else:
                pretty_ownermap += '.'
        pretty_ownermap = ' '.join(pretty_ownermap.rstrip())
        pretty_board = ['%s   %s' % (brow, orow[2:]) for brow, orow in zip(pretty_board, pretty_ownermap.split("\n")[1:])]
    print("\n".join(pretty_board), file=f)
    print('    ' + ' '.join(colstr[:N]), file=f)
    print('', file=f)


def dump_subtree(node, thres=N_SIMS/50, indent=0, f=sys.stderr, recurse=True):
    """ print this node and all its children with v >= thres. """
    print("%s+- %s %.3f (%d/%d, prior %d/%d, rave %d/%d=%.3f, pred %.3f)" %
          (indent*' ', str_coord(node.pos.last), node.winrate(),
           node.w, node.v, node.pw, node.pv, node.aw, node.av,
           float(node.aw)/node.av if node.av > 0 else float('nan'),
           float(-net.predict_winrate(node.pos) + 1) / 2), file=f)
    if not recurse:
        return
    for child in sorted(node.children, key=lambda n: n.v, reverse=True):
        if child.v >= thres:
            dump_subtree(child, thres=thres, indent=indent+3, f=f)


def print_tree_summary(tree, sims, f=sys.stderr):
    best_nodes = sorted(tree.children, key=lambda n: n.v, reverse=True)[:5]
    best_seq = []
    node = tree
    while node is not None:
        best_seq.append(node.pos.last)
        node = node.best_move()
    best_predwinrate = float(-net.predict_winrate(best_nodes[0].pos) + 1) / 2
    print('[%4d] winrate %.3f/%.3f | seq %s | can %s' %
          (sims, best_nodes[0].winrate(), best_predwinrate, ' '.join([str_coord(c) for c in best_seq[1:6]]),
           ' '.join(['%s(%.3f|%d/%.3f)' % (str_coord(n.pos.last), n.winrate(), n.v, n.prior()) for n in best_nodes])), file=f)


def parse_coord(s):
    if s == 'pass':
        return None
    return W+1 + (N - int(s[1:])) * W + colstr.index(s[0].upper())


def str_coord(c):
    if c is None:
        return 'pass'
    row, col = divmod(c - (W+1), W)
    return '%c%d' % (colstr[col], N - row)


# various main programs

def play_and_train(batches_per_game=4, disp=False):
    positions = []

    tree = TreeNode(pos=empty_position())
    tree.expand()
    owner_map = W*W*[0]
    while True:
        owner_map = W*W*[0]
        next_tree = tree_search(tree, N_SIMS, owner_map)

        distribution = np.zeros((N, N))
        for child in tree.children:
            if child.pos.last is None:
                continue  # TODO pass moves
            p = float(child.v) / tree.v
            c = child.pos.last
            x, y = c % W - 1, c // W - 1
            distribution[y, x] = p
        positions.append((tree.pos, distribution))

        tree = next_tree
        print_pos(tree.pos, sys.stdout, owner_map)

        if tree.pos.last is None and tree.pos.last2 is None:
            score = 1 if tree.pos.score() > 0 else -1
            if tree.pos.n % 2:
                score = -score
            print('Two passes, score: B%+.1f' % (score,))

            count = tree.pos.score()
            if tree.pos.n % 2:
                count = -count
            print('Counted score: B%+.1f' % (count,))
            break
        if float(tree.w)/tree.v < RESIGN_THRES and tree.v > N_SIMS / 10:
            score = 1  # win for player to-play from this position
            if tree.pos.n % 2:
                score = -score
            print('Resign (%d), score: B%+.1f' % (tree.pos.n % 2, score))

            count = tree.pos.score()
            if tree.pos.n % 2:
                count = -count
            print('Counted score: B%+.1f' % (count,))
            break
        if tree.pos.n > N*N*2:
            print('Stopping too long a game.')
            score = 0
            break

    # score here is for black to play (player-to-play from empty_position)
    print(score)
    if disp:
        dump_subtree(tree)
    for i in range(batches_per_game):
        net.fit_game(positions, score)

    # fit flipped positions
    def flip_vert(board):
        return '\n'.join(reversed(board[:-1].split('\n'))) + ' '
    for i in range(batches_per_game):
        net.fit_game(positions, score, board_transform=flip_vert)

    def flip_horiz(board):
        return '\n'.join([' ' + l[1:][::-1] for l in board.split('\n')])
    for i in range(batches_per_game):
        net.fit_game(positions, score, board_transform=flip_horiz)

    def flip_both(board):
        return '\n'.join(reversed([' ' + l[1:][::-1] for l in board[:-1].split('\n')])) + ' '
    for i in range(batches_per_game):
        net.fit_game(positions, score, board_transform=flip_both)

    # TODO 90\deg rot


def selfplay(snapshot_interval=100, disp=True):
    i = 0
    while True:
        print('Self-play of game #%d ...' % (i,))
        play_and_train(disp=disp)
        i += 1
        if i % snapshot_interval == 0:
            weights_fname = '%s_%09d.weights.h5' % (net.model_name, i)
            print(weights_fname)
            net.model.save_weights(weights_fname)


def game_io(computer_black=False):
    """ A simple minimalistic text mode UI. """

    tree = TreeNode(pos=empty_position())
    tree.expand()
    owner_map = W*W*[0]
    while True:
        if not (tree.pos.n == 0 and computer_black):
            print_pos(tree.pos, sys.stdout, owner_map)

            sc = raw_input("Your move: ")
            c = parse_coord(sc)
            if c is not None:
                # Not a pass
                if tree.pos.board[c] != '.':
                    print('Bad move (not empty point)')
                    continue

                # Find the next node in the game tree and proceed there
                nodes = filter(lambda n: n.pos.last == c, tree.children)
                if not nodes:
                    print('Bad move (rule violation)')
                    continue
                tree = nodes[0]

            else:
                # Pass move
                if tree.children[0].pos.last is None:
                    tree = tree.children[0]
                else:
                    tree = TreeNode(pos=tree.pos.pass_move())

            print_pos(tree.pos)

        owner_map = W*W*[0]
        tree = tree_search(tree, N_SIMS, owner_map)
        if tree.pos.last is None and tree.pos.last2 is None:
            score = tree.pos.score()
            if tree.pos.n % 2:
                score = -score
            print('Game over, score: B%+.1f' % (score,))
            break
        if float(tree.w)/tree.v < RESIGN_THRES:
            print('I resign.')
            break
    print('Thank you for the game!')


def gtp_io():
    """ GTP interface for our program.  We can play only on the board size
    which is configured (N), and we ignore color information and assume
    alternating play! """
    known_commands = ['boardsize', 'clear_board', 'komi', 'play', 'genmove',
                      'final_score', 'quit', 'name', 'version', 'known_command',
                      'list_commands', 'protocol_version', 'tsdebug']

    tree = TreeNode(pos=empty_position())
    tree.expand()

    while True:
        try:
            line = raw_input().strip()
        except EOFError:
            break
        if line == '':
            continue
        command = [s.lower() for s in line.split()]
        if re.match('\d+', command[0]):
            cmdid = command[0]
            command = command[1:]
        else:
            cmdid = ''
        owner_map = W*W*[0]
        ret = ''
        if command[0] == "boardsize":
            if int(command[1]) != N:
                print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
                ret = None
        elif command[0] == "clear_board":
            tree = TreeNode(pos=empty_position())
            tree.expand()
        elif command[0] == "komi":
            # XXX: can we do this nicer?!
            tree.pos = Position(board=tree.pos.board, cap=(tree.pos.cap[0], tree.pos.cap[1]),
                                n=tree.pos.n, ko=tree.pos.ko, last=tree.pos.last, last2=tree.pos.last2,
                                komi=float(command[1]))
        elif command[0] == "play":
            c = parse_coord(command[2])
            if c is not None:
                # Find the next node in the game tree and proceed there
                if tree.children is not None and filter(lambda n: n.pos.last == c, tree.children):
                    tree = filter(lambda n: n.pos.last == c, tree.children)[0]
                else:
                    # Several play commands in row, eye-filling move, etc.
                    tree = TreeNode(pos=tree.pos.move(c))

            else:
                # Pass move
                if tree.children[0].pos.last is None:
                    tree = tree.children[0]
                else:
                    tree = TreeNode(pos=tree.pos.pass_move())
        elif command[0] == "genmove":
            tree = tree_search(tree, N_SIMS, owner_map)
            if tree.pos.last is None:
                ret = 'pass'
            elif float(tree.w)/tree.v < RESIGN_THRES:
                ret = 'resign'
            else:
                ret = str_coord(tree.pos.last)
        elif command[0] == "final_score":
            score = tree.pos.score()
            if tree.pos.n % 2:
                score = -score
            if score == 0:
                ret = '0'
            elif score > 0:
                ret = 'B+%.1f' % (score,)
            elif score < 0:
                ret = 'W+%.1f' % (-score,)
        elif command[0] == "name":
            ret = 'michi'
        elif command[0] == "version":
            ret = 'simple go program demo'
        elif command[0] == "tsdebug":
            print_pos(tree_search(tree, N_SIMS, W*W*[0], disp=True))
        elif command[0] == "list_commands":
            ret = '\n'.join(known_commands)
        elif command[0] == "known_command":
            ret = 'true' if command[1] in known_commands else 'false'
        elif command[0] == "protocol_version":
            ret = '2'
        elif command[0] == "quit":
            print('=%s \n\n' % (cmdid,), end='')
            break
        else:
            print('Warning: Ignoring unknown command - %s' % (line,), file=sys.stderr)
            ret = None

        print_pos(tree.pos, sys.stderr, owner_map)
        if ret is not None:
            print('=%s %s\n\n' % (cmdid, ret,), end='')
        else:
            print('?%s ???\n\n' % (cmdid,), end='')
        sys.stdout.flush()


if __name__ == "__main__":
    global net
    net = AGZeroModel()
    net.create()
    if len(sys.argv) < 2:
        # Default action
        game_io()
    elif sys.argv[1] == "white":
        game_io(computer_black=True)
    elif sys.argv[1] == "gtp":
        gtp_io()
    elif sys.argv[1] == "tsbenchmark":
        t_start = time.time()
        print_pos(tree_search(TreeNode(pos=empty_position()), N_SIMS, W*W*[0], disp=False).pos)
        print('Tree search with %d playouts took %.3fs with %d threads; speed is %.3f playouts/thread/s' %
              (N_SIMS, time.time() - t_start, multiprocessing.cpu_count(),
               N_SIMS / ((time.time() - t_start) * multiprocessing.cpu_count())))
    elif sys.argv[1] == "tsdebug":
        print_pos(tree_search(TreeNode(pos=empty_position()), N_SIMS, W*W*[0], disp=True).pos)
    elif sys.argv[1] == "selfplay":
        selfplay()
    else:
        print('Unknown action', file=sys.stderr)
