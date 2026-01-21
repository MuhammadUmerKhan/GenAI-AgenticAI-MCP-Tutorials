import sqlite3
from typing import Dict, Any, Optional


class KnowledgeBase:
    """
    Manages the local SQLite Knowledge Base for crypto metadata and cached API data.
    Supports creation, insertion, querying, and updating of coin records.
    """

    def __init__(self, db_path: str = DB_PATH):
        """
        Initializes the KnowledgeBase with the given database path.

        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the database table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS coins (
                coin TEXT PRIMARY KEY,
                symbol TEXT,
                launch_year INTEGER,
                consensus TEXT,
                chain_type TEXT,
                last_price REAL,
                market_cap REAL,
                price_timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_coin(self, coin_data: Dict[str, Any]):
        """
        Adds or replaces a coin record in the KB.

        :param coin_data: Dictionary with coin details (must include 'coin' key).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO coins VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            coin_data.get('coin'),
            coin_data.get('symbol'),
            coin_data.get('launch_year'),
            coin_data.get('consensus'),
            coin_data.get('chain_type'),
            coin_data.get('last_price'),
            coin_data.get('market_cap'),
            coin_data.get('price_timestamp')
        ))
        conn.commit()
        conn.close()

    def search_coin(self, coin: str, field: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Searches for a coin's data in the KB. Returns full record or specific field.

        :param coin: Coin name (e.g., "Bitcoin").
        :param field: Optional specific field to return (e.g., "last_price").
        :return: Dictionary of coin data or None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if field:
            cursor.execute(f'SELECT {field} FROM coins WHERE LOWER(coin) = LOWER(?)', (coin,))
            result = cursor.fetchone()
            conn.close()
            return {field: result[0]} if result else None
        else:
            cursor.execute('SELECT * FROM coins WHERE LOWER(coin) = LOWER(?)', (coin,))
            result = cursor.fetchone()
            conn.close()
            if result:
                return {
                    'coin': result[0], 'symbol': result[1], 'launch_year': result[2],
                    'consensus': result[3], 'chain_type': result[4], 'last_price': result[5],
                    'market_cap': result[6], 'price_timestamp': result[7]
                }
            return None

    def update_coin(self, coin: str, updates: Dict[str, Any]):
        """
        Updates specific fields for a coin in the KB.

        :param coin: Coin name.
        :param updates: Dictionary of fields to update (e.g., {'last_price': 50000.0}).
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        set_clause = ', '.join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [coin]
        cursor.execute(f'UPDATE coins SET {set_clause} WHERE coin = ?', values)
        conn.commit()
        conn.close()