"""Tiny logger: prints a line and keeps it so load() can save the run log."""
RUN_LOG = []


def step(message):
    print(message)
    RUN_LOG.append(message)


def reset():
    RUN_LOG.clear()
