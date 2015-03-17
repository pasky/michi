Michi --- Minimalistic Go MCTS Engine
=====================================

Michi aims to be a minimalistic but full-fledged Computer Go program based
on state-of-art methods (Monte Carlo Tree Search) and written in Python.
Our goal is to make it easier for new people to enter the domain of
Computer Go, peek under the hood of a "real" playing engine and be able
to learn by hassle-free experiments - with the algorithms, add heuristics,
etc.

Our target size is under 500 lines of code (without user interface, tables
and empty lines / comments).  Currently, it can often win against GNUGo
on 9x9 on an old i3 4-thread notebook.  This is not meant to be a competitive
engine; simplicity and clear code is preferred over optimization (after all,
it's in Python!).

Michi is distributed under the MIT licence.  Now go forth, hack and peruse!

Usage
-----

If you want to try it out, just start the script.  You can also pass the
gtp argument and start it in gogui, or let it play GNUGo:

	gogui/bin/gogui-twogtp -black './michi.py gtp' -white 'gnugo --mode=gtp --chinese-rules --capture-all-dead' -size 9 -komi 7.5 -verbose -auto

It is *highly* recommended that you download Pachi large-scale pattern files
(patterns.prob, patterns.spat):

	http://pachi.or.cz/pat/gogod-handikgspachi/

Store them in the current directory for Michi to find.

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
  * Simple time management.
  * gogui visualization support.
  * Group/liberty tracking in the board position implementation.
  * Probability distribution playouts.
  * Some popular variant of the last-good-reply policy.
  * Most crucial heuristics: Better ko fight support, throwin detection,
    nakade support, ...
  * Diminishing RAVE by origin move number and other fancy algorithmics.
