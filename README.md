Michi --- Minimalistic Go MCTS Engine
=====================================

Michi aims to be a minimalistic but full-fledged Computer Go program based
on state-of-art methods (Monte Carlo Tree Search) and written in Python.
Our goal is to make it easier for new people to enter the domain of
Computer Go, peek under the hood of a "real" playing engine and be able
to learn by hassle-free experiments - with the algorithms, add heuristics,
etc.

The algorithm code size is 540 lines of code (without user interface, tables
and empty lines / comments).  Currently, it can often win against GNUGo
on 9×9 on an old i3 notebook, be about even with GNUGo on 15×15 on a modern
higher end computer and about two stones weaker on 19×19 (spending no more
than 30s per move).

This is not meant to be a competitive engine; simplicity and clear code is
preferred over optimization (after all, it's in Python!).  But compared to
other minimalistic engines, this one should be able to beat beginner
intermediate human players, and I believe that a *fast* implementation
of exactly the same heuristics would be around 4k KGS or even better.

Michi is distributed under the MIT licence.  Now go forth, hack and peruse!

Usage
-----

If you want to try it out, just start the script.  You can also pass the
gtp argument and start it in gogui, or let it play GNUGo:

	gogui/bin/gogui-twogtp -black './michi.py gtp' -white 'gnugo --mode=gtp --chinese-rules --capture-all-dead' -size 9 -komi 7.5 -verbose -auto

It is *highly* recommended that you download Michi large-scale pattern files
(patterns.prob, patterns.spat):

	http://pachi.or.cz/michi-pat/

Store and unpack them in the current directory for Michi to find.

Understanding and Hacking
-------------------------

Note that while all strong Computer Go programs currently use the MCTS
algorithm, there are many particular variants, differing particularly
regarding the playout policy mechanics and the way tree node priors
are constructed and incorporated (this is not about individual heuristics
but the way they are integrated).  Michi uses the MCTS flavor used in
Pachi and Fuego, but e.g. Zen and CrazyStone take quite a different
approach to this.  For a general introduction to Michi-style MCTS algorithm,
see Petr Baudis' Master Thesis http://pasky.or.cz/go/prace.pdf, esp.
Sec. 2.1 to 2.3 and Sec. 3.3 to 3.4.

The ethymology of Michi is "Minimalistic Pachi".  If you would like
to try your hands at hacking a competitive Computer Go engine, try Pachi! :-)
Michi has been inspired by Sunfish - a minimalistic chess engine.  Sadly
(or happily?), for computers Go is a lot more complicated than chess, even
if you want to just implement the rules.

We would like to encourage you to experiment with the heuristics and try
to add more.  But please realize that if some heuristic seems to work well,
you should verify how it works in a more competitive engine (in conjunction
with other heuristics and many more playouts) and play-test it on at least
a few hundred games with a reference opponent (not just the program without
the heuristic, self-play testing greatly exaggerates any improvements).

TODO
----

Strong Computer Go programs tend to accumulate many specialized,
sophisticated, individually low-yield heuristics.  These are mostly
out of scope of Michi in order to keep things short and simple.
However, other than that, there are certainly things that Michi should
or could contain, roughly in order of priority:

  * Superko support.
  * Support for early passing and GTP stone status protocol.
  * gogui visualization support.
  * Group/liberty tracking in the board position implementation.

If you would like to increase the strength of the program, the lowest
hanging fruit is likely:

  * Tune parameters using Rémi Coulom's CLOP.
  * Simple time management.  (See the Pachi paper.)
  * Pondering (search game tree during opponent's move) support.
  * Make it faster - either by optimizations (see group tracking above)
    or 1:1 rewrite in a faster language.
  * Two/three liberty semeai reading in playouts.  (See also CFG patterns.)
  * Tsumego improvements - allow single-stone selfatari only for throwins
    and detect nakade shapes.
  * Try true probability distribution playouts + Rémi Coulom's MM patterns.

(Most of the mistakes Michi makes is caused by the awfully low number of
playouts; I believe that with 20× speedup, which seems very realistic, the same
algorithm could easily get to KGS 4k on 19×19.  One of the things I would hope
to inspire is rewrite of the same algorithm in different, faster programming
languages; hopefully seeing a done real-world thing is easier than developing
it from scratch.  What about a Go engine in the Go language?)

**michi-c** is such a rewrite of Michi, in plain C.  It seems to play even with
GNUGo when given 3.3s/move: https://github.com/db3108/michi-c2.

Note: there is a clone version of the michi python code (slower than michi-c2) 
that is available at https://github.com/db3108/michi-c. 
This simpler version can be read in parallel with the michi.py python code.

**michi-go** is a rewrite of Michi in the Go language:
https://github.com/traveller42/michi-go
