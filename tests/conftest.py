import sys
from pathlib import Path

# Make the project root importable from tests without sys.path hacks in each file
sys.path.insert(0, str(Path(__file__).parent.parent))
