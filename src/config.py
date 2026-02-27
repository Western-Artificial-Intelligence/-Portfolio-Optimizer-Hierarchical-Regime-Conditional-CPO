"""
Central configuration for the Portfolio Optimizer project.
All paths, date ranges, and asset universe definitions live here.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results2"
GNN_RESULTS_DIR = PROJECT_ROOT / "results5"  # GNN v2 outputs (fixed alpha floor model)
GNN_RESULTS_DIR_V1 = PROJECT_ROOT / "results3"  # GNN v1 outputs (kept for reference)

STOCK_HISTORY_CSV = DATA_DIR / "stock_history.csv"
STOCK_PROFILES_CSV = DATA_DIR / "stock_profiles.csv"
ECONOMIC_INDICATORS_CSV = DATA_DIR / "economic_indicators.csv"
YIELD_CURVE_CSV = DATA_DIR / "yield_curve_spread.csv"

# Ensure results dirs exist
RESULTS_DIR.mkdir(exist_ok=True)
GNN_RESULTS_DIR.mkdir(exist_ok=True)
GNN_RESULTS_DIR_V1.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Date ranges (from paper §5)
# ──────────────────────────────────────────────
TRAIN_START = "2010-01-01"
TRAIN_END = "2018-12-31"
VAL_START = "2019-01-01"
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-12-31"

# ──────────────────────────────────────────────
# Asset universe
# ──────────────────────────────────────────────
BENCHMARK = "SPY US Equity"

# Canadian tickers (from Bloomberg data)
CANADIAN_TICKERS = [
    # Financials
    "RY CN Equity",    # Royal Bank
    "TD CN Equity",    # TD Bank
    "BMO CN Equity",   # BMO
    "BNS CN Equity",   # Scotiabank
    # Energy
    "CNQ CN Equity",   # Canadian Natural Resources
    "SU CN Equity",    # Suncor
    "ENB CN Equity",   # Enbridge
    "TRP CN Equity",   # TC Energy
    "CCO CN Equity",   # Cameco (uranium/energy)
    # Industrials
    "CNR CN Equity",   # CN Rail
    "CP CN Equity",    # CP Rail
    "WCN CN Equity",   # Waste Connections
    # Telecom
    "BCE CN Equity",   # BCE
    "T CN Equity",     # Telus
    # Utilities
    "FTS CN Equity",   # Fortis
    # Materials / Gold
    "AEM CN Equity",   # Agnico Eagle
    "ABX CN Equity",   # Barrick Gold
    "IVN CN Equity",   # Ivanhoe Mines
    "TCL/A CN Equity", # Transcontinental
    # Tech
    "SHOP CN Equity",  # Shopify
    "CSU CN Equity",   # Constellation Software
    "CLS CN Equity",   # Celestica
    "GIB/A CN Equity", # CGI
    "OTEX CN Equity",  # OpenText
    "DSG CN Equity",   # Descartes
    "KXS CN Equity",   # Kinaxis
    # Consumer
    "ATD CN Equity",   # Alimentation Couche-Tard
    "DOL CN Equity",   # Dollarama
    "DOO CN Equity",   # BRP
    "MG CN Equity",    # Magna
    "QSR CN Equity",   # Restaurant Brands
]

ALL_TICKERS = CANADIAN_TICKERS + [BENCHMARK]

# ──────────────────────────────────────────────
# Bloomberg field names (from stock_history.csv)
# ──────────────────────────────────────────────
PRICE_FIELD = "PX_LAST"
TOTAL_RETURN_FIELD = "TOT_RETURN_INDEX_GROSS_DVDS"
VOLUME_FIELD = "PX_VOLUME"
MARKET_CAP_FIELD = "CUR_MKT_CAP"
PE_RATIO_FIELD = "PE_RATIO"

# ──────────────────────────────────────────────
# Portfolio constraints (from paper §3.1)
# ──────────────────────────────────────────────
MAX_WEIGHT = 0.15        # No single stock > 15%
MIN_WEIGHT = 0.0         # Long-only
TRANSACTION_COST_BPS = 10  # 10 bps round-trip

# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_SIZE = (14, 8)
