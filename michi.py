#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The below is currently not true:
## This is the main Michi module that implements the complete search logic.
## The Go board implementation can be found in board.py.
## This file is not executable - it needs to be used through a user interface
## frontend like gtpmichi.py or textmichi.py.

from collections import namedtuple
from itertools import count
import math
from operator import itemgetter
import random
import re


# Given a board of size NxN (N=9, 19, ...), we represent the position
# as an (N+1)*(N+2) string, with '.' (empty), 'X' (to-play player),
# 'x' (other player), and whitespace (off-board border to make rules
# implementation easier).  Coordinates are just indices in this string.
N = 9
W = N + 2
empty = "\n".join([(N+1)*' '] + N*[' '+N*'.'] + [(N+2)*' '])
colstr = 'ABCDEFGHJKLMNOPQRST'
MAX_GAME_LEN = N * N * 3

UCB1_C = 0.1
EXPAND_VISITS = 2
PRIOR_EVEN = 3
PRIOR_CAPTURE = 5
PRIOR_PAT3 = 5
REPORT_PERIOD = 20
RESIGN_THRES = 0.2

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
    # XXX: Store entry neighbor so we don't revisit it
    # XXX: Shoot beams to speed things up
    while fringe:
        c = fringe.pop()
        for d in neighbors(c):
            if board[d] == p:
                board = board_put(board, d, '#')
                fringe.append(d)
    return board


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
    return contact_res[p].search(board)

def libpos(board):
    """ yield a liberty of #-marked group """
    m = contact(board, '.')
    if not m:
        return None
    if m.group(0)[0] == '.':
        c = m.start()
    else:
        c = m.end() - 1
    return c


def in_atari(board, c):
    """ return None if not in atari, the liberty coordinate otherwise """
    fboard = floodfill(board, c)
    l = libpos(fboard)
    if l is None:
        return None
    # Ok, any other liberty?
    fboard = board_put(fboard, l, 'L')
    if libpos(fboard) is not None:
        return None
    return l


def is_eyeish(board, c):
    """ test if c is inside a single-color diamong and return the diamond
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


class Position(namedtuple('Position', 'board cap n ko last last2')):
    """ Implementation of simple Chinese Go rules;
    n is how many moves were played so far """

    def print_board(self):
        if self.n % 2 == 0:  # to-play is black
            board = self.board.replace('x', 'O')
            Xcap, Ocap = self.cap
        else:  # to-play is white
            board = self.board.replace('X', 'O').replace('x', 'X')
            Ocap, Xcap = self.cap
        print('Move: %-3d   Black: %d caps   White: %d caps' % (self.n, Xcap, Ocap))
        pretty_board = ' '.join(board.rstrip())
        if self.last is not None:
            pretty_board = pretty_board[:self.last*2-1] + '(' + board[self.last] + ')' + pretty_board[self.last*2+2:]
        rowcounter = count()
        pretty_board = "\n".join([' %-02d%s' % (N-i, row[2:]) for row, i in zip(pretty_board.split("\n")[1:], rowcounter)])
        print(pretty_board)
        print('    ' + ' '.join(colstr[:N]))
        print('')

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
            if contact(fboard, '.'):
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
        if not contact(floodfill(board, c), '.'):
            return None

        # Update the position and return
        return Position(board=board.swapcase(), cap=(self.cap[1], capX), n=self.n + 1, ko=ko, last=c, last2=self.last)

    def pass_move(self):
        """ pass - i.e. return a flipped position """
        return Position(board=self.board.swapcase(), cap=(self.cap[1], self.cap[0]), n=self.n + 1, ko=None, last=None, last2=self.last)

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
        """ generate a randomly shuffled list of moves surrounding the last
        two moves (but with the last move having priority) """
        dlist = []
        for c in self.last, self.last2:
            if c is None:  continue
            dlist += list(neighbors(c) + diag_neighbors(c))
        random.shuffle(dlist)
        return dlist

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
            touches_X = contact(fboard, 'X')
            touches_x = contact(fboard, 'x')
            if touches_X and not touches_x:
                board = fboard.replace('#', 'X')
            elif touches_x and not touches_X:
                board = fboard.replace('#', 'x')
            else:
                board = fboard.replace('#', ':')  # seki, rare
        return board.count('X') - board.count('x')


def empty_position():
    """ Return an initial board position """
    return Position(board=empty, cap=(0, 0), n=0, ko=None, last=None, last2=None)


# pattern routines

def pat_rot90(p):
    return [p[2][0]+p[1][0]+p[0][0], p[2][1]+p[1][1]+p[0][1], p[2][2]+p[1][2]+p[0][2]]
def pat_vertflip(p):
    return [p[2], p[1], p[0]]
def pat_horizflip(p):
    return [l[::-1] for l in p]
def pat_swapcolors(p):
    return [l.replace('X', 'Z').replace('x', 'z').replace('O', 'X').replace('o', 'x').replace('Z', 'O').replace('z', 'o') for l in p]
def pat_regex(p):
    return ''.join([l.replace('.', '\\.').replace('?', '.').replace('x', '[^X]').replace('O', 'x').replace('o', '[^x]') for l in p])
# A gigantic regex that will match the 3x3 pattern on move neighborhood
# strings, accounting for all possible transpositions
patternre_src = [pat_regex(p) for p in patternsrc for p in [p, pat_rot90(p)] for p in [p, pat_vertflip(p)] for p in [p, pat_horizflip(p)] for p in [p, pat_swapcolors(p)]]
patternre = re.compile('|'.join(patternre_src))

def neighborhood(board, c):
    return board[c-W-1 : c-W+2] + board[c-1 : c+2] + board[c+W-1 : c+W+2]


# montecarlo playout policy

def gen_playout_moves(pos):
    """ Yield candidate next moves in the order of preference; this is one
    of the main places where heuristics dwell, try adding more! """
    local_moves = pos.last_moves_neighbors()

    # Check whether any local group is in atari and fill that liberty
    for c in local_moves:
        if pos.board[c] in 'Xx':
            d = in_atari(pos.board, c)
            if d is not None:
                yield (d, 'capture')

    # Try to apply a 3x3 pattern on the local neighborhood
    for c in local_moves:
        if patternre.match(neighborhood(pos.board, c)):
            yield (c, 'pat3')

    # Try *all* available moves, but starting from a random point
    # (in other words, play a random move)
    x, y = random.randint(1, N), random.randint(1, N)
    for c in pos.moves(y*W + x):
        yield (c, 'random')


def mcplayout(pos, disp=False):
    """ Start a Monte Carlo playout from a given position, return score
    for to-play player at the starting position"""
    start_n = pos.n
    passes = 0
    while passes < 2 and pos.n < MAX_GAME_LEN:
        if disp:  pos.print_board()

        pos2 = None
        for c, kind in gen_playout_moves(pos):
            pos2 = pos.move(c)
            if pos2 is not None:
                break
        if pos2 is None:  # no valid moves, pass
            pos = pos.pass_move()
            passes += 1
            continue
        passes = 0
        pos = pos2
    score = pos.score()
    if start_n % 2 == pos.n % 2:
        return score
    else:
        return -score

def mcbenchmark(n):
    """ run n Monte-Carlo playouts from empty position, return avg. score """
    sumscore = 0
    for i in range(0, n):
        sumscore += mcplayout(empty_position())
    return float(sumscore) / n


# Monte Carlo tree, grown asymmetrically

class TreeNode():
    """ Monte-Carlo tree node;
    v is #visits, w is #wins for to-play (expected reward is w/v)
    pv, pw are prior values (node value = w/v + pw/pv)
    children is None for leaf nodes """
    def __init__(self, pos):
        self.pos = pos
        self.v = 0
        self.w = 0
        self.pv = PRIOR_EVEN
        self.pw = PRIOR_EVEN
        self.children = None

    def expand(self):
        """ add and initialize children to a leaf node """
        self.children = []
        for c, kind in gen_playout_moves(self.pos):
            pos2 = self.pos.move(c)
            if pos2 is None:
                continue
            node = TreeNode(pos2)
            self.children.append(node)

            # Add some priors to bias search towards more sensible moves
            # Note that there are many other ways to incorporate the priors
            if kind == 'capture':
                node.pv += PRIOR_CAPTURE
                node.pw += PRIOR_CAPTURE
            elif kind == 'pat3':
                node.pv += PRIOR_PAT3
                node.pw += PRIOR_PAT3

        if not self.children:
            # No possible moves, add a pass move
            self.children.append(TreeNode(self.pos.pass_move()))

    def ucb1_urgency(self, n0):
        return float(self.w+self.pw)/(self.v+self.pv) + UCB1_C * math.sqrt(2*math.log(n0) / (self.v+1))

    def child_values(self):
        return [float(node.w+1)/(node.v+1) for node in self.children]

    def best_move(self):
        """ best move is the most simulated one """
        return self.children[max(enumerate([node.v for node in self.children]), key=itemgetter(1))[0]]


def str_tree_summary(tree, sims):
    best_nodes = [tree.children[i] for i, u in sorted(enumerate(tree.child_values()), key=itemgetter(1), reverse=True)[:5]]
    best_seq = []
    node = tree
    while node is not None:
        best_seq.append(node.pos.last)
        if node.children is None:
            break
        node = node.best_move()
    return ('[%4d] winrate %.3f | best %s | seq %s' %
            (sims, 1.0-float(tree.w)/tree.v,
             ' '.join(['%s(%.3f)' % (str_coord(n.pos.last), float(n.w+1)/(n.v+1)) for n in best_nodes]),
             ' '.join([str_coord(c) for c in best_seq[1:5]]),
             ))


def tree_search(tree, n, disp=False):
    """ Perform MCTS search from a given position for a given #iterations """
    # Initialize root node
    if tree.children is None:
        tree.expand()

    for i in range(n):
        # Descend to the leaf
        tree.v += 1
        nodes = [tree]
        passes = 0
        while nodes[-1].children is not None and passes < 2:
            if disp:  nodes[-1].pos.print_board()
            # Pick the most urgent child
            urgencies = [node.ucb1_urgency(nodes[-1].v) for node in nodes[-1].children]
            if disp:
                print(', '.join(['%s:%d/%d:%.3f' % (str_coord(nodes[-1].children[ci].pos.last), nodes[-1].children[ci].w, nodes[-1].children[ci].v, u)
                                 for ci, u in enumerate(urgencies)]))
            ci, u = max(enumerate(urgencies), key=itemgetter(1))
            nodes.append(nodes[-1].children[ci])
            if nodes[-1].pos.last is None:
                passes += 1
            else:
                passes = 0
            nodes[-1].v += 1
            if nodes[-1].children is None and nodes[-1].v >= EXPAND_VISITS:
                nodes[-1].expand()

        # Run a simulation
        if disp:  print('** SIMULATION **')
        score = mcplayout(nodes[-1].pos, disp=disp)

        # Back-propagate the result
        for node in reversed(nodes):
            if disp:  print('updating', str_coord(node.pos.last), score < 0)
            node.w += score < 0  # score is for to-play, node statistics for just-played
            score = -score

        if i > 0 and i % REPORT_PERIOD == 0:
            print(str_tree_summary(tree, i))

    print(str_tree_summary(tree, n))
    return tree.best_move()


def parse_coord(s):
    if s == 'pass':
        return None
    return W+1 + (N - int(s[1])) * W + colstr.index(s[0])


def str_coord(c):
    if c is None:
        return 'pass'
    row, col = divmod(c - (W+1), W)
    return '%c%d' % (colstr[col], N - row)


def game_io():
    """ A simple UI for playing on the board, no move generation involved;
    intended for testing. """

    tree = TreeNode(pos=empty_position())
    tree.expand()
    while True:
        tree.pos.print_board()

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
        tree = tree_search(tree, 500)
        if tree.pos.last is None and tree.pos.last2 is None:
            print('Game over, score: B%d' % (tree.pos.score(),))
            break
        if float(tree.w)/tree.v < RESIGN_THRES:
            print('I resign.')
            break
    print('Thank you for the game!')


if __name__ == "__main__":
    game_io()
    # print(mcplayout(empty_position(), disp=True))
    # print(mcbenchmark(20))
    # tree_search(TreeNode(pos=empty_position()), 1000).pos.print_board()
