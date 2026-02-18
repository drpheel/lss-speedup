"""Compatibility entrypoint for live LSS camera/BEV streaming.

The implementation now lives in `src.streaming` modules.
"""

from src.streaming.app import main


if __name__ == "__main__":
    main()
