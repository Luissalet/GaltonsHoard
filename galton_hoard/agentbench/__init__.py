"""Agent bench: real agent harnesses on published agentic benchmarks, with one model.

Each cell of a campaign is one benchmark x one harness. A cell runs as a job of the container trial runner (``harbor``): every task gets its own
Linux container, the harness is installed in it and works on the task until it stops, then the task's own hidden tests decide. Galton stores one row
per trial and reads the cells as pass rates with intervals, per benchmark and per harness, apart from the trials whose verifier never ran.

Modules:

* ``spec``: harness lines (from the settings) and the benchmark catalogue, with a deterministic task subset and a dev/test split per benchmark;
* ``harbor``: the command line of a job and the reading of a finished job folder into trial rows;
* ``scoring``: cells, ranking per benchmark, paired comparison of one harness against each other one on the same tasks;
* ``store``: campaigns and trials in Galton's database;
* ``runner``: a campaign as a stored job that keeps a fixed number of trials in flight.
"""
