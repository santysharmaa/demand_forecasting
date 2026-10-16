"""Shared settings. Change values here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SEED = 42
N_STORES = 10
N_ITEMS = 20
START, END = "2022-01-01", "2025-12-31"

HORIZON = 7            # we forecast 7 days ahead -> every lag must be >= 7
VALID_DAYS = 60        # used to tune / early stop
TEST_DAYS = 60         # final untouched test period

# Inventory assumptions (these are ASSUMPTIONS - say so in your README/interview)
LEAD_TIME_DAYS = 7
SERVICE_Z = 1.65       # ~95% service level
ORDER_COST = 50.0      # cost of placing one order
HOLDING_RATE = 0.20    # yearly holding cost = 20% of unit cost
