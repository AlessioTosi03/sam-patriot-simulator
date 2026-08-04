"""Top-level launcher preserved as a thin wrapper.

Import and run `main.run()` to preserve the original
single-file entrypoint behavior.
"""

from main import run


if __name__ == '__main__':
    run()