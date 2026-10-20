import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
WEB_DATA = ROOT / "web" / "assets" / "data"
for p in (DATA, REPORTS, FIG, WEB_DATA):
    p.mkdir(parents=True, exist_ok=True)
SEED = 42
