from fastmcp import FastMCP
from pathlib import Path
import sqlite3

# Proper paths
DB_PATH = Path(__file__).parent / "database" / "expense_tracker.db"
CATEGORIES_PATH = Path(__file__).parent / "database" / "categories.json"

mcp = FastMCP(name="Expense Tracker")

def init_db():
    # Ensure database directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as c:
        c.execute("PRAGMA foreign_keys = ON")
        c.execute("PRAGMA journal_mode = WAL")
        c.execute("PRAGMA synchronous = NORMAL")
        c.execute("PRAGMA cache_size = -64000")  # 64MB cache

        c.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL CHECK(date GLOB '????-??-??'),
                amount      REAL NOT NULL CHECK(amount >= 0),
                currency    TEXT NOT NULL DEFAULT 'USD',
                category    TEXT NOT NULL DEFAULT '',
                subcategory TEXT NOT NULL DEFAULT '',
                note        TEXT NOT NULL DEFAULT ''
            );
        """)
        c.commit()

# Initialize database on import
init_db()

@mcp.tool
def add_expense(date: str, amount: float, category: str = "", subcategory: str = "", note: str = "") -> dict:
    """Add a new expense to the database."""
    with sqlite3.connect(DB_PATH) as c:
        c.execute("PRAGMA synchronous = NORMAL")  # Good for each connection
        cur = c.execute(
            "INSERT INTO expenses (date, amount, category, subcategory, note) VALUES (?, ?, ?, ?, ?)",
            (date, amount, category, subcategory, note)
        )
        c.commit()
        return {"status": "success", "id": cur.lastrowid}

@mcp.tool
def list_expense(start_date: str = "", end_date: str = "") -> list[dict]:
    """List all expenses from the database ordered by date."""
    # Default to full range if dates are empty
    if not start_date:
        start_date = "0000-01-01"
    if not end_date:
        end_date = "9999-12-31"

    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC
            """,
            (start_date, end_date)
        )
        cols = [description[0] for description in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]

@mcp.tool
def summarize(start_date: str, end_date: str, category: str | None = None) -> list[dict]:
    """Summarize expenses by category within an inclusive date range."""
    query = """
        SELECT category, SUM(amount) AS total_amount
        FROM expenses
        WHERE date BETWEEN ? AND ?
    """
    params = [start_date, end_date]

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " GROUP BY category ORDER BY total_amount DESC"

    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(query, params)
        cols = [description[0] for description in cur.description]
        rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]

@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    """Return the categories JSON file content."""
    if not CATEGORIES_PATH.exists():
        # Return empty valid JSON if file doesn't exist yet
        return '{"categories": []}'
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
    # for running the server: fastmcp run 01_practice_mcp_server.py --transport http --host 0.0.0.0 --port 8000 or uv run 01_practice_mcp_server.py
    # In debugging mode: uv run fastmcp dev 01_practice_mcp_server.py
    
    # For Deployment visit fastmcp.cloud and deploy the server