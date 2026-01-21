# Constants
DB_PATH = "../database/crypto_kb.db"
STALE_MINUTES = 5  # Data older than this is considered stale
REJECTION_MESSAGE = "INSUFFICIENT DATA – Not found in Knowledge Base or API"
DISALLOWED_INTENTS = ["prediction", "advice", "hypothetical"]
API_BASE_URL = "https://api.freecryptoapi.com/v1/"