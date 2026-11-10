import sys
from pathlib import Path

# Add integration directory to sys.path so `app` can be imported
integration_dir = Path(__file__).resolve().parent.parent
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))
