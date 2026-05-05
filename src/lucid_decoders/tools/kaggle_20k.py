"""Backward-compatible entrypoint for older Kaggle notebook cells.

The active Kaggle helper now targets the 15k run.
"""

from lucid_decoders.tools.kaggle_15k import *  # noqa: F401,F403
from lucid_decoders.tools.kaggle_15k import main


if __name__ == "__main__":
    main()
