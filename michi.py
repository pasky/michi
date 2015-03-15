#!/usr/bin/env pypy
# -*- coding: utf-8 -*-
# A minimalistic Go-playing engine attempting to strike a balance between
# brevity, educational value and strength.  It can beat GNUGo on 9x9 board
# on a modest 4-thread computer.

# FIXME: No superko support.  This is a big bug, of course.

from __future__ import print_function
from collections import namedtuple
from itertools import count
import math
import multiprocessing
from multiprocessing.pool import Pool
from operator import itemgetter
import random
import re
import sys


# Given a board of size NxN (N=9, 19, ...), we represent the position
# as an (N+1)*(N+2) string, with '.' (empty), 'X' (to-play player),
# 'x' (other player), and whitespace (off-board border to make rules
# implementation easier).  Coordinates are just indices in this string.
N = 9
W = N + 2
empty = "\n".join([(N+1)*' '] + N*[' '+N*'.'] + [(N+2)*' '])
colstr = 'ABCDEFGHJKLMNOPQRST'
MAX_GAME_LEN = N * N * 3

N_SIMS = 2000
UCB1_C = 0.1
RAVE_EQUIV = 3000
EXPAND_VISITS = 2
PRIOR_EVEN = 6  # should be even number; 0.5 prior
PRIOR_SELFATARI = 10  # negative prior
PRIOR_CAPTURE = 10
PRIOR_PAT3 = 10
PRIOR_CFG = [20, 15, 10]  # priors for moves in cfg dist. 1, 2, 3
PRIOR_EMPTYAREA = 10
REPORT_PERIOD = 200
PROB_SSAREJECT = 0.9  # probability of rejecting suggested self-atari in playout
PROB_RSAREJECT = 0.5  # probability of rejecting random self-atari in playout; this is lower than above to allow nakade
RESIGN_THRES = 0.2
FASTPLAY20_THRES = 0.8  # if at 20% playouts winrate is >this, stop reading
FASTPLAY5_THRES = 0.95  # if at 5% playouts winrate is >this, stop reading

patternsrc = [  # 3x3 playout patterns; X,O are colors, x,o are their inverses
       ["XOX",  # hane pattern - enclosing hane
        "...",
        "???"],
       ["XO.",  # hane pattern - non-cutting hane
        "...",
        "?.?"],
       ["XO?",  # hane pattern - magari
        "X..",
        "x.?"],
       # ["XOO",  # hane pattern - thin hane
       #  "...",
       #  "?.?", "X",  - only for the X player
       [".O.",  # generic pattern - katatsuke or diagonal attachment; similar to magari
        "X..",
        "..."],
       ["XO?",  # cut1 pattern (kiri] - unprotected cut
        "O.o",
        "?o?"],
       ["XO?",  # cut1 pattern (kiri] - peeped cut
        "O.X",
        "???"],
       ["?X?",  # cut2 pattern (de]
        "O.O",
        "ooo"],
       ["OX?",  # cut keima
        "o.O",
        "???"],
       ["X.?",  # side pattern - chase
        "O.?",
        "  ?"],
       ["OX?",  # side pattern - block side cut
        "X.O",
        "   "],
       ["?X?",  # side pattern - block side connection
        "x.O",
        "   "],
       ["?XO",  # side pattern - sagari
        "x.x",
        "   "],
       ["?OX",  # side pattern - cut
        "X.O",
        "   "],
       ]


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
    # XXX: Use bytearray to speed things up?
    p = board[c]
    board = board_put(board, c, '#')
    fringe = [c]
    # XXX: Shoot beams to speed things up
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if board[d] == p:
                board = board_put(board, d, '#')
                fringe.append(d)
    return board


# Regex that matches various kind of points adjecent to '#' (floodfilled) points
contact_res = dict()
for p in ['.', 'x', 'X']:
    rp = '\\.' if p == '.' else p
    contact_res_src = [
        '#' + rp,  # p at right
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
    if m.group(0)[0] == p:
        return m.start()
    else:
        return m.end() - 1


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
            fboard = floodfill(board, d)  # board with the adjecent group replaced by '#'
            if contact(fboard, '.') is not None:
                continue  # some liberties left
            capcount = fboard.count('#')
            if capcount == 1:
                singlecaps.append(d)
            capX += capcount
            board = fboard.replace('#', '.')  # capture the group
        # Set ko
        if in_enemy_eye and len(singlecaps) == 1:
            ko = singlecaps[0]
        else:
            ko = None
        # Test for suicide
        if contact(floodfill(board, c), '.') is None:
            return None

        # Update the position and return
        return Position(board=board.swapcase(), cap=(self.cap[1], capX),
                n=self.n + 1, ko=ko, last=c, last2=self.last, komi=self.komi)

    def pass_move(self):
        """ pass - i.e. return a flipped position """
        return Position(board=self.board.swapcase(), cap=(self.cap[1], self.cap[0]),
                n=self.n + 1, ko=None, last=None, last2=self.last, komi=self.komi)

    def moves(self, i0):
        """ Generate a list of moves (includes false positives - suicide moves;
        does not include true-eye-filling moves), starting from a given board
        index (that can be used for randomization)"""
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
        priority)"""
        clist = []
        for c in self.last, self.last2:
            if c is None:  continue
            dlist = [c] + list(neighbors(c) + diag_neighbors(c))
            random.shuffle(dlist)
            clist += dlist
        return clist

    def score(self):
        """ compute score for to-play player; this assumes a final position
        with all dead stones captured"""
        board = self.board
        i = 0
        while True:
            i = self.board.find('.', i+1)
            if i == -1:
                break
            fboard = floodfill(board, i)
            touches_X = contact(fboard, 'X') is not None
            touches_x = contact(fboard, 'x') is not None
            if touches_X and not touches_x:
                board = fboard.replace('#', 'X')
            elif touches_x and not touches_X:
                board = fboard.replace('#', 'x')
            else:
                board = fboard.replace('#', ':')  # seki, rare
        komi = self.komi if self.n % 2 == 1 else -self.komi
        return board.count('X') - board.count('x') + komi

    def print_board(self, f=sys.stderr):
        if self.n % 2 == 0:  # to-play is black
            board = self.board.replace('x', 'O')
            Xcap, Ocap = self.cap
        else:  # to-play is white
            board = self.board.replace('X', 'O').replace('x', 'X')
            Ocap, Xcap = self.cap
        print('Move: %-3d   Black: %d caps   White: %d caps  Komi: %.1f' % (self.n, Xcap, Ocap, self.komi), file=f)
        pretty_board = ' '.join(board.rstrip())
        if self.last is not None:
            pretty_board = pretty_board[:self.last*2-1] + '(' + board[self.last] + ')' + pretty_board[self.last*2+2:]
        rowcounter = count()
        pretty_board = "\n".join([' %-02d%s' % (N-i, row[2:]) for row, i in zip(pretty_board.split("\n")[1:], rowcounter)])
        print(pretty_board, file=f)
        print('    ' + ' '.join(colstr[:N]), file=f)
        print('', file=f)


def empty_position():
    """ Return an initial board position """
    return Position(board=empty, cap=(0, 0), n=0, ko=None, last=None, last2=None, komi=7.5)


###############
# go heuristics

def fix_atari(pos, c, singlept_ok=False):
    """ An atari/capture analysis routine that checks the group at c,
    determining whether (i) it is in atari (ii) if it can escape it,
    either by playing on its liberty or counter-capturing another group.

    The return value is a tuple of (boolean, coord), indicating whether
    the group is in atari and how to escape/capture (or None if impossible).

    singlept_ok means that we will not try to save one-point groups (ko) """

    fboard = floodfill(pos.board, c)
    if singlept_ok and fboard.count('#') == 1:
        return (False, None)
    # Find a liberty
    l = contact(fboard, '.')
    # Ok, any other liberty?
    fboard = board_put(fboard, l, 'L')
    if contact(fboard, '.') is not None:
        return (False, None)
    # In atari! If it's the opponent's group, that's enough...
    if pos.board[c] == 'x':
        return (True, l)

    # Before thinking about defense, what about counter-capturing
    # a neighboring group?
    ccboard = fboard
    while True:
        othergroup = contact(ccboard, 'x')
        if othergroup is None:
            break
        a, ccl = fix_atari(pos, othergroup)
        if ccl is not None:
            return (True, ccl)
        # XXX: floodfill is better for big groups
        ccboard = board_put(ccboard, othergroup, '%')

    # We are escaping.  Will playing our last liberty gain
    # at least two liberties?  Re-floodfill to account for connecting
    escpos = pos.move(l)
    if escpos is None:
        return (True, None)  # oops, suicidal move
    fboard = floodfill(escpos.board, l)
    l_new = contact(fboard, '.')
    fboard = board_put(fboard, l_new, 'L')
    # print(str_coord(l_new), fboard, file=sys.stderr)
    if contact(fboard, '.') is not None:
        return (True, l)  # good, there is still some liberty remaining

    return (True, None)


def cfg_distances(board, c):
    """ return a board map listing common fate graph distances from
    a given point - this corresponds to the concept of locality while
    contracting groups to single points """
    cfg_map = W*W*[-1]
    cfg_map[c] = 0

    fringe = [c]
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if board[d].isspace() or cfg_map[d] >= 0:
                continue
            if board[d] != '.' and board[d] == board[c]:
                cfg_map[d] = cfg_map[c]
            else:
                cfg_map[d] = cfg_map[c] + 1
            fringe.append(d)
    return cfg_map


def line_height(c):
    """ Return the line number above nearest board edge """
    row, col = divmod(c - (W+1), W)
    return min(row, col, N-1-row, N-1-col)


def empty_area(board, c, dist=3):
    """ Check whether there are any stones in Manhattan distance up
    to 3 """
    for d in neighbors(c):
        if board[d] in 'Xx':
            return False
        elif board[d] == '.' and dist > 1 and not empty_area(board, d, dist-1):
            return False
    return True


# pattern routines

def pat_expand(pat):
    """ All possible neighborhood configurations matching a given pattern """
    def pat_rot90(p):
        return [p[2][0] + p[1][0] + p[0][0], p[2][1] + p[1][1] + p[0][1], p[2][2] + p[1][2] + p[0][2]]
    def pat_vertflip(p):
        return [p[2], p[1], p[0]]
    def pat_horizflip(p):
        return [l[::-1] for l in p]
    def pat_swapcolors(p):
        return [l.replace('X', 'Z').replace('x', 'z').replace('O', 'X').replace('o', 'x').replace('Z', 'O').replace('z', 'o') for l in p]
    def pat_wildexp(p, c, to):
        i = p.find(c)
        if i == -1:
            return [p]
        return reduce(lambda a, b: a + b, [pat_wildexp(p[:i] + t + p[i+1:], c, to) for t in to])
    def pat_wildcards(pat):
        return [p for p in pat_wildexp(pat, '?', list('.XO '))
                  for p in pat_wildexp(p, 'x', list('.O '))
                  for p in pat_wildexp(p, 'o', list('.X '))]
    return [p for p in [pat, pat_rot90(pat)]
              for p in [p, pat_vertflip(p)]
              for p in [p, pat_horizflip(p)]
              for p in [p, pat_swapcolors(p)]
              for p in pat_wildcards(''.join(p))]

pat3set = set([p.replace('O', 'x') for p in patternsrc for p in pat_expand(p)])

def neighborhood(board, c):
    return (board[c-W-1 : c-W+2] + board[c-1 : c+2] + board[c+W-1 : c+W+2]).replace('\n', ' ')


###########################
# montecarlo playout policy

def gen_playout_moves(pos, heuristic_set):
    """ Yield candidate next moves in the order of preference; this is one
    of the main places where heuristics dwell, try adding more!

    heuristic_set is the set of coordinates considered for applying heuristics;
    this is the immediate neighborhood of last two moves in the playout, but
    the whole board while prioring the tree. """

    # Check whether any local group is in atari and fill that liberty
    # print('local moves', [str_coord(c) for c in heuristic_set], file=sys.stderr)
    already_suggested = set()
    for c in heuristic_set:
        if pos.board[c] in 'Xx':
            in_atari, d = fix_atari(pos, c)
            if d is not None and d not in already_suggested:
                yield (d, 'capture')
                already_suggested.add(d)

    # Try to apply a 3x3 pattern on the local neighborhood
    already_suggested = set()
    for c in heuristic_set:
        if pos.board[c] == '.' and c not in already_suggested and neighborhood(pos.board, c) in pat3set:
            yield (c, 'pat3')
            already_suggested.add(c)

    # Try *all* available moves, but starting from a random point
    # (in other words, play a random move)
    x, y = random.randint(1, N), random.randint(1, N)
    for c in pos.moves(y*W + x):
        yield (c, 'random')


def mcplayout(pos, amaf_map, disp=False):
    """ Start a Monte Carlo playout from a given position,
    return score for to-play player at the starting position;
    amaf_map is board-sized scratchpad recording who played at a given
    position first """
    if disp:  print('** SIMULATION **', file=sys.stderr)
    start_n = pos.n
    passes = 0
    while passes < 2 and pos.n < MAX_GAME_LEN:
        if disp:  pos.print_board()

        pos2 = None
        # We simply try the moves our heuristics generate, in a particular
        # order.  This is called "rule-based playouts" and is easier to do,
        # but the strongest programs use "probability distribution playouts"
        # which use a more flexible approach to move selection.
        for c, kind in gen_playout_moves(pos, pos.last_moves_neighbors()):
            if disp and kind != 'random':
                print('move suggestion', str_coord(c), kind, file=sys.stderr)
            pos2 = pos.move(c)
            if pos2 is None:
                continue
            if random.random() <= (PROB_RSAREJECT if kind == 'random' else PROB_SSAREJECT):
                in_atari, d = fix_atari(pos2, c, singlept_ok=True)
                if in_atari:
                    if disp:  print('rejecting self-atari move', str_coord(c), file=sys.stderr)
                    pos2 = None
                    continue
            if amaf_map[c] == 0:  # Mark the coordinate with 1 for black
                amaf_map[c] = 1 if pos.n % 2 == 0 else -1
            break
        if pos2 is None:  # no valid moves, pass
            pos = pos.pass_move()
            passes += 1
            continue
        passes = 0
        pos = pos2
    score = pos.score()
    if disp:  print('** SCORE B%+.1f **' % (score if pos.n % 2 == 0 else -score), file=sys.stderr)
    if start_n % 2 == pos.n % 2:
        return score, amaf_map
    else:
        return -score, amaf_map

def mcbenchmark(n):
    """ run n Monte-Carlo playouts from empty position, return avg. score """
    sumscore = 0
    for i in range(0, n):
        sumscore += mcplayout(empty_position(), W*W*[0])[0]
    return float(sumscore) / n


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
        self.pv = PRIOR_EVEN
        self.pw = PRIOR_EVEN/2
        self.av = 0
        self.aw = 0
        self.children = None

    def expand(self):
        """ add and initialize children to a leaf node """
        if self.pos.last is not None:
            cfg_map = cfg_distances(self.pos.board, self.pos.last)
        else:
            cfg_map = None
        self.children = []
        childset = dict()
        for c, kind in gen_playout_moves(self.pos, range(N, (N+1)*W)):
            pos2 = self.pos.move(c)
            if pos2 is None:
                continue
            try:
                node = childset[pos2.last]
            except KeyError:
                node = TreeNode(pos2)
                self.children.append(node)
                childset[pos2.last] = node

            # Add some priors to bias search towards more sensible moves
            # Note that there are many other ways to incorporate the priors
            if kind == 'capture':
                node.pv += PRIOR_CAPTURE
                node.pw += PRIOR_CAPTURE
            elif kind == 'pat3':
                node.pv += PRIOR_PAT3
                node.pw += PRIOR_PAT3

            if cfg_map is not None:
                assert cfg_map[node.pos.last] > 0
                if cfg_map[node.pos.last]-1 < len(PRIOR_CFG):
                    node.pv += PRIOR_CFG[cfg_map[node.pos.last]-1]
                    node.pw += PRIOR_CFG[cfg_map[node.pos.last]-1]

            height = line_height(c)  # 0-index
            if height <= 2 and empty_area(self.pos.board, c):
                # No stones around; negative prior for 1st + 2nd line, positive
                # for 3rd line; sanitizes opening and invasions
                if height <= 1:
                    node.pv += PRIOR_EMPTYAREA
                    node.pw += 0
                if height == 2:
                    node.pv += PRIOR_EMPTYAREA
                    node.pw += PRIOR_EMPTYAREA

            in_atari, d = fix_atari(pos2, c, singlept_ok=True)
            if in_atari:
                node.pv += PRIOR_SELFATARI
                node.pw += 0  # negative prior

        if not self.children:
            # No possible moves, add a pass move
            self.children.append(TreeNode(self.pos.pass_move()))
            return

    def ucb1_urgency(self, n0):
        expectation = float(self.w+self.pw)/(self.v+self.pv)
        return expectation + UCB1_C * math.sqrt(2*math.log(n0) / (self.v+1))

    def rave_urgency(self):
        v = self.v + self.pv
        expectation = float(self.w+self.pw) / v
        if self.av == 0:
            return expectation
        rave_expectation = float(self.aw) / self.av
        beta = self.av / (self.av + v + float(v) * self.av / RAVE_EQUIV)
        return beta * rave_expectation + (1-beta) * expectation

    def winrate(self):
        if self.v == 0:
            return float('nan')
        return float(self.w) / self.v

    def best_move(self):
        """ best move is the most simulated one """
        return self.children[max(enumerate([node.v for node in self.children]), key=itemgetter(1))[0]]

    def dump_subtree(self, thres=N_SIMS/50, indent=0, f=sys.stderr, recurse=True):
        """ print this node and all its children with v >= thres. """
        print("%s+- %s %.3f (%d/%d, prior %d/%d, rave %d/%d=%.3f, urgency %.3f)" %
              (indent*' ', str_coord(self.pos.last), self.winrate(),
               self.w, self.v, self.pw, self.pv, self.aw, self.av,
               float(self.aw)/self.av if self.av > 0 else float('nan'),
               self.rave_urgency()), file=f)
        if not recurse:
            return
        for child in sorted(self.children, key=lambda n: n.v, reverse=True):
            if child.v >= thres:
                child.dump_subtree(thres=thres, indent=indent+3, f=f)


def str_tree_summary(tree, sims):
    best_nodes = [tree.children[i] for i, u in sorted(enumerate([n.v for n in tree.children]), key=itemgetter(1), reverse=True)[:5]]
    best_seq = []
    node = tree
    while node is not None:
        best_seq.append(node.pos.last)
        if node.children is None:
            break
        node = node.best_move()
    return ('[%4d] winrate %.3f | seq %s | can %s' %
            (sims, best_nodes[0].winrate(),
             ' '.join([str_coord(c) for c in best_seq[1:6]]),
             ' '.join(['%s(%.3f)' % (str_coord(n.pos.last), n.winrate()) for n in best_nodes]),
             ))


def tree_descend(tree, amaf_map, disp=False):
    """ Descend through the tree to a leaf """
    tree.v += 1
    nodes = [tree]
    passes = 0
    while nodes[-1].children is not None and passes < 2:
        if disp:  nodes[-1].pos.print_board()

        # Pick the most urgent child
        # urgencies = list(enumerate([node.ucb1_urgency(nodes[-1].v) for node in nodes[-1].children]))
        urgencies = list(enumerate([node.rave_urgency() for node in nodes[-1].children]))
        if disp:
            for c in nodes[-1].children:
                c.dump_subtree(recurse=False)
        random.shuffle(urgencies)  # randomize the max in case of equal urgency
        ci, u = max(urgencies, key=itemgetter(1))

        nodes.append(nodes[-1].children[ci])

        c = nodes[-1].pos.last
        if disp:  print('chosen %s' % (str_coord(c),), file=sys.stderr)
        if c is None:
            passes += 1
        else:
            passes = 0
            if amaf_map[c] == 0:  # Mark the coordinate with 1 for black
                amaf_map[c] = 1 if nodes[-2].pos.n % 2 == 0 else -1

        nodes[-1].v += 1  # updating visits on the way down represents "virtual loss", relevant for parallelization
        if nodes[-1].children is None and nodes[-1].v >= EXPAND_VISITS:
            nodes[-1].expand()

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


pool = None

def tree_search(tree, n, disp=False):
    """ Perform MCTS search from a given position for a given #iterations """
    # Initialize root node
    if tree.children is None:
        tree.expand()

    # We could simply run tree_descend(), mcplayout(), tree_update()
    # sequentially in a loop.  However, we have an easy (though not optimal)
    # way to parallelize by distributing the mcplayout() calls to other
    # processes using the multiprocessing module.  mcplayout() consumes
    # maybe more than 90% CPU, especially on larger boards.

    n_workers = multiprocessing.cpu_count() if not disp else 1  # set to 1 when debugging
    global pool
    if pool is None:
        pool = Pool(processes=n_workers)
    ongoing = []
    i = 0
    while i < n:
        if len(ongoing) >= n_workers:
            # Too many playouts running? Wait a bit...
            ongoing[0][0].wait(0.01)
        else:
            i += 1
            if i > 0 and i % REPORT_PERIOD == 0:
                print(str_tree_summary(tree, i), file=sys.stderr)

            # Descend the tree
            amaf_map = W*W*[0]
            nodes = tree_descend(tree, amaf_map, disp=disp)

            # Issue an mcplayout job to the worker pool
            ongoing.append((pool.apply_async(mcplayout, (nodes[-1].pos, amaf_map, disp)), nodes))

        # Any playouts are finished yet?
        for job, nodes in ongoing:
            if not job.ready():
                continue
            # Yes! Store it in the tree.
            score, amaf_map = job.get()
            tree_update(nodes, amaf_map, score, disp=disp)
            ongoing.remove((job, nodes))

        # Early stop test
        best_wr = tree.best_move().winrate()
        if i > n*0.05 and best_wr > FASTPLAY5_THRES or i > n*0.2 and best_wr > FASTPLAY20_THRES:
            break

    tree.dump_subtree()
    print(str_tree_summary(tree, i), file=sys.stderr)
    return tree.best_move()


###################
# user interface(s)

def parse_coord(s):
    if s == 'pass':
        return None
    return W+1 + (N - int(s[1:])) * W + colstr.index(s[0].upper())


def str_coord(c):
    if c is None:
        return 'pass'
    row, col = divmod(c - (W+1), W)
    return '%c%d' % (colstr[col], N - row)


def game_io(computer_black=False):
    """ A simple UI for playing on the board, no move generation involved;
    intended for testing. """

    tree = TreeNode(pos=empty_position())
    tree.expand()
    while True:
        if not (tree.pos.n == 0 and computer_black):
            tree.pos.print_board(sys.stdout)

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

            tree.pos.print_board()

        tree = tree_search(tree, N_SIMS)
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
    tree = TreeNode(pos=empty_position())
    tree.expand()

    for line in sys.stdin:
        line = line.rstrip()
        command = [s.lower() for s in line.split()]
        if re.match('\d+', command[0]):
            cmdid = command[0]
            command = command[1:]
        else:
            cmdid = ''
        ret = ''
        if command[0] == "boardsize":
            if int(command[1]) != N:
                print("Warning: Trying to set incompatible boardsize %s (!= %d)" % (command[1], N), file=sys.stderr)
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
                # Not a pass
                if tree.pos.board[c] != '.':
                    print('Bad move (not empty point)')
                    continue

                # Find the next node in the game tree and proceed there
                if tree.children is None:
                    tree.expand()  # Triggers in case of several plays in row
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
        elif command[0] == "genmove":
            tree = tree_search(tree, N_SIMS)
            if tree.pos.last is None and tree.pos.last2 is None:
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
            tree_search(tree, N_SIMS, disp=True).pos.print_board()
        elif command[0] == "list_commands":
            ret = '\n'.join(['boardsize', 'clear_board', 'komi', 'play', 'genmove', 'final_score', 'name', 'version', 'list_commands', 'ts_debug'])
        elif command[0] == "protocol_version":
            ret = '2'
        else:
            print('Warning: Ignoring unknown command - %s' % (line,), file=sys.stderr)
            ret = None

        tree.pos.print_board(sys.stderr)
        if ret is not None:
            print('=%s %s\n\n' % (cmdid, ret,), end='')
        else:
            print('?%s ???\n\n' % (cmdid,), end='')
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default action
        game_io()
    elif sys.argv[1] == "white":
        game_io(computer_black=True)
    elif sys.argv[1] == "gtp":
        gtp_io()
    elif sys.argv[1] == "mcdebug":
        print(mcplayout(empty_position(), W*W*[0], disp=True)[0])
    elif sys.argv[1] == "mcbenchmark":
        print(mcbenchmark(20))
    elif sys.argv[1] == "tsbenchmark":
        tree_search(TreeNode(pos=empty_position()), N_SIMS, disp=False).pos.print_board()
    elif sys.argv[1] == "tsdebug":
        tree_search(TreeNode(pos=empty_position()), N_SIMS, disp=True).pos.print_board()
    else:
        print('Unknown action', file=sys.stderr)
