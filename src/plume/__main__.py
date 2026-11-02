"""Allow ``python -m plume``."""
import sys

from plume.cli import main

if __name__ == "__main__":
    sys.exit(main())
