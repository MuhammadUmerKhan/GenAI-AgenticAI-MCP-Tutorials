import os

# Constants
DB_PATH = "../database/crypto_kb.db"
STALE_MINUTES = 5  # Data older than this is considered stale
REJECTION_MESSAGE = "INSUFFICIENT DATA – Not found in Knowledge Base or API"
DISALLOWED_INTENTS = ["prediction", "advice", "hypothetical"]
API_BASE_URL = "https://api.freecryptoapi.com/v1/"

CRYPTO_API_TOKEN = os.getenv("FREE_CRYPTO_API_KEY")

headers = {
    "Authorization": f"Bearer {CRYPTO_API_TOKEN}",
}