"""Top-level launcher preserved as a thin wrapper.

This file used to contain both the simulation and visualization code.
Those parts have been moved into `environment.py`, `visualization.py`,
and `main.py`. Import and run `main.run()` to preserve the original
single-file entrypoint behavior.
"""

from main import run


if __name__ == '__main__':
    run()