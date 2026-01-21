import sqlite3, json
from datetime import datetime, UTC
from typing import Dict, Any, Optional
from config import DB_PATH

class KnowledgeBase:
    """
    Manages the local SQLite Knowledge Base for crypto price data.
    Stores fields that match the FreeCryptoAPI response + fetch_timestamp for freshness.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Creates the coins table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coins (
                coin                    TEXT PRIMARY KEY,   -- e.g. "Bitcoin" (not from API)
                symbol                  TEXT NOT NULL,
                lowest_price            REAL,
                highest_price           REAL,
                last_price              REAL,
                daily_change_percentage REAL,
                date                    TEXT,               -- API-provided date string
                source_exchange         TEXT,
                fetch_timestamp         TEXT                -- ISO UTC when we fetched
            )
        ''')
        conn.commit()
        conn.close()

    def add_coin(self, coin_data: Dict[str, Any]):
        """
        INSERT or REPLACE a full coin record.
        coin_data should contain 'coin' + fields from API or manual entry.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO coins (
                coin, symbol, lowest_price, highest_price, last_price,
                daily_change_percentage, date, source_exchange, fetch_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            coin_data.get('coin'),
            coin_data.get('symbol'),
            coin_data.get('lowest_price'),
            coin_data.get('highest_price'),
            coin_data.get('last_price'),
            coin_data.get('daily_change_percentage'),
            coin_data.get('date'),
            coin_data.get('source_exchange'),
            coin_data.get('fetch_timestamp')
        ))
        conn.commit()
        conn.close()

    def search_coin(self, coin: str, field: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Search by coin name (case-insensitive).
        Returns full record or single field.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if field:
            cursor.execute(
                f'SELECT {field} FROM coins WHERE LOWER(coin) = LOWER(?)',
                (coin,)
            )
            result = cursor.fetchone()
            conn.close()
            return {field: result[0]} if result else None
        else:
            cursor.execute(
                'SELECT * FROM coins WHERE LOWER(coin) = LOWER(?)',
                (coin,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    'coin': row[0],
                    'symbol': row[1],
                    'lowest_price': row[2],
                    'highest_price': row[3],
                    'last_price': row[4],
                    'daily_change_percentage': row[5],
                    'date': row[6],
                    'source_exchange': row[7],
                    'fetch_timestamp': row[8],
                }
            return None

    def update_coin(self, coin: str, updates: Dict[str, Any]):
        """
        Update selected fields for a given coin.
        """
        if not updates:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        set_clause = ', '.join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [coin]
        cursor.execute(f'UPDATE coins SET {set_clause} WHERE coin = ?', values)
        conn.commit()
        conn.close()

    def is_stale(self, coin: str, max_age_minutes: int = 5) -> bool:
        """
        Convenience method: check if data is older than max_age_minutes.
        Uses fetch_timestamp (ISO format).
        """
        data = self.search_coin(coin)
        if not data or not data.get('fetch_timestamp'):
            return True

        try:
            ts = datetime.fromisoformat(data['fetch_timestamp'].replace('Z', '+00:00'))
            age = datetime.now(UTC) - ts
            return age.total_seconds() > (max_age_minutes * 60)
        except (ValueError, TypeError):
            return True

# ────────────────────────────────────────────────
# Example usage / test
# ────────────────────────────────────────────────
if __name__ == "__main__":
    kb = KnowledgeBase()

    # Example: insert fresh API-like data
    api_response_example = {
        "symbol": "BTC",
        "last_price": "89851.53",
        "last_btc": "1",
        "lowest_price": "87932.07",
        "highest_price": "90099.5",
        "date": "2026-01-21 09:29:30",
        "daily_change_percentage": "0.16809224003144",
        "source_exchange": "binance",
        "fetch_timestamp": datetime.now(UTC).isoformat()
    } 

    # if api_response_example:
    #     full_record = {
    #         "coin": "Bitcoin",               # ← you provide this
    #         **api_response_example           # spread API fields
    #     }
    #     kb.add_coin(full_record)

    # Or update only some fields
    # kb.update_coin("Bitcoin", {"last_price": 90500.0, "fetch_timestamp": datetime.now(UTC).isoformat()})

    print(json.dumps(kb.search_coin("Bitcoin"), indent=2, default=str))