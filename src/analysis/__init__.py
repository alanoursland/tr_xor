import sys
from pathlib import Path

# Setup import path once for the whole package
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))