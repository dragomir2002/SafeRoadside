import sys
from pathlib import Path

# The detector modules live one level up and are not an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
