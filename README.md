Michi --- Minimalistic Go MCTS Engine
=====================================

Michi aims to be a minimalistic but full-fledged Computer Go program based
on state-of-art methods (Monte Carlo Tree Search) and written in Python.
Our goal is to make it easier for new people to enter the domain of
Computer Go, see under the hood of a real playing engine and be able
to learn by hassle-free experiments - with the algorithms, add heuristics,
etc.

Our target size is under 300 lines of code (without user interface and
empty lines / comments).  We would like to aim at 4k KGS 19x19 strength,
though this might require some very beefy hardware as the board engine
in Python is pretty slow (it could be at least 5x optimized with some
extra work, though).  This is not meant to be a competitive engine;
simplicity and clear code is preferred over optimization (after all,
it's in Python!).

Note that while all strong Computer Go programs currently use the MCTS
algorithm, there are many particular variants, differing particularly
regarding the playout policy mechanics and the way tree node priors
are constructed and incorporated (this is not about individual heuristics
but the way they are integrated).  Michi uses the MCTS flavor used in
Pachi and Fuego, but e.g. Zen and CrazyStone take quite a different
approach to this.

The ethymology of Michi is "Minimalistic Pachi".  If you would like
to try your hands at hacking a competitive Computer Go engine, try Pachi! :-)
Michi has been inspired by Sunfish - a minimalistic chess engine.

Michi is distributed under the MIT licence.  Now go forth, hack and peruse!

...

Or not.  This is an early stage prototype that does not meet the goals
above at all.  Work in progress...
