from config import API_BASE_URL, headers
from datetime import datetime, UTC
from typing import Dict, Any, Optional
import requests, json

def call_crypto_api(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Calls FreeCryptoAPI /getData endpoint and extracts relevant fields.
    
    Returns a flat dict with the fields we care about + our fetch timestamp.
    Returns None on failure or invalid response.
    
    Note: This function no longer takes 'data_type' because the endpoint
    returns all relevant price fields at once.
    """
    url = f"{API_BASE_URL}getData?symbol={symbol.upper()}"
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            return None

        symbols = data.get("symbols", [])
        if not symbols:
            return None

        # Take first item (we requested one symbol)
        item = symbols[0]

        return {
            "symbol": item.get("symbol"),
            "last_price": float(item.get("last", 0)),
            "lowest_price": float(item.get("lowest", 0)),
            "highest_price": float(item.get("highest", 0)),
            "daily_change_percentage": float(item.get("daily_change_percentage", 0)),
            "date": item.get("date"),
            "source_exchange": item.get("source_exchange"),
            "fetch_timestamp": datetime.now(UTC).isoformat()
        }

    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None
    
if __name__ == "__main__":
    response = call_crypto_api("BTC")
    print(json.dumps(response, indent=2, default=str))