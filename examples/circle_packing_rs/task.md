# Goal

- Place n = 26 circles inside the unit square so that the sum of their radii is as large as possible.
- No two circles may overlap, and every circle must lie entirely inside the square.
- The best known sum of radii for 26 circles is about 2.635.

# How the program is judged

The program prints one line per circle, `x y r`, 26 lines in total. The evaluator checks the constraints
itself and scores the sum of radii; a layout that violates a constraint is incorrect and scores nothing.
The program must be a single Rust file that compiles with `rustc -O` using the standard library only, and
must finish within 30 seconds.
