#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The below is currently not true:
## This is the main Michi module that implements the complete search logic.
## The Go board implementation can be found in board.py.
## This file is not executable - it needs to be used through a user interface
## frontend like gtpmichi.py or textmichi.py.

from collections import namedtuple
from itertools import count
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
        print('Move: % 3d   Black: %d caps   White: %d caps' % (self.n, Xcap, Ocap))
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
    of the main places where heuristics dwell """
    local_moves = pos.last_moves_neighbors()

    # Check whether any local group is in atari and fill that liberty
    for c in local_moves:
        if pos.board[c] in 'Xx':
            d = in_atari(pos.board, c)
            if d is not None:
                yield d

    # Try to apply a 3x3 pattern on the local neighborhood
    for c in local_moves:
        if patternre.match(neighborhood(pos.board, c)):
            yield c

    # Try *all* available moves, but starting from a random point
    # (in other words, play a random move)
    x, y = random.randint(1, N), random.randint(1, N)
    for c in pos.moves(y*W + x):
        yield c


def mcplayout(pos, disp=False):
    """ Start a Monte Carlo playout from a given position, return score
    for to-play player at the starting position"""
    start_n = pos.n
    passes = 0
    while passes < 2 and pos.n < MAX_GAME_LEN:
        if disp:  pos.print_board()

        pos2 = None
        for c in gen_playout_moves(pos):
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


def parse_coord(s):
    return W+1 + (N - int(s[1])) * W + colstr.index(s[0])


def str_coord(c):
    row, col = divmod(c - (W+1), W)
    return '%c%d' % (colstr[col], N - row)


def game_io():
    """ A simple UI for playing on the board, no move generation involved;
    intended for testing. """

    pos = empty_position()
    while True:
        pos.print_board()
        sc = raw_input("Your move: ")
        c = parse_coord(sc)
        if pos.board[c] != '.':
            print('Bad move (not empty point)')
            continue
        pos2 = pos.move(c)
        if pos2 is None:
            print('Bad move (rule violation)')
            continue
        pos = pos2


if __name__ == "__main__":
    # game_io()
    # print(mcplayout(empty_position(), disp=True))
    print(mcbenchmark(20))
