"""
Configuration - edit ONLY these fields for each new brand.
"""

# ── Brand Settings ──────────────────────────────────────────────────────────
BRAND_NAME   = "Walmart"
APP_NAME     = BRAND_NAME
KEYWORDS     = ["Walmart store", "Walmart supercenter"]

# ── App Store (leave blank - auto-discovery will skip cleanly) ──────────────
APP_STORE_ID = ""
APP_COUNTRY  = "us"

# ── Platform Branding ─────────────────────────────────────────────────────────
PLATFORM_TITLE    = "Walmart Intelligence Platform"
PLATFORM_SUBTITLE = "Customer Insights & Operations"
PLATFORM_ICON     = "🛒"

# ── AI Model ──────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Scraper Settings ──────────────────────────────────────────────────────────
MAX_REVIEW_PAGES = 10

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR       = "data"
REVIEWS_CSV    = "data/reviews.csv"
BUSINESSES_CSV = "data/businesses.csv"

# ── Analytics Settings ────────────────────────────────────────────────────────
ANOMALY_THRESHOLD_STARS = 0.4
SIGNIFICANT_DELTA_STARS = 0.15
