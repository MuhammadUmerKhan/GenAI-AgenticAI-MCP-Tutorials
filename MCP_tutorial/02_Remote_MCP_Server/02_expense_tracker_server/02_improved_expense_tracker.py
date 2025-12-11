
from fastmcp import FastMCP
import os
import aiosqlite
import asyncio

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "expense_tracker.db")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "database", "categories.json")

mcp = FastMCP(name="Expense Tracker")

async def init_db():
    async with aiosqlite.connect(DB_PATH) as c:
        await c.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT DEFAULT '',
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            );
        """)
        await c.commit()

# Smart initialization that works in both cases:
#  - When the script is run directly (no loop yet) → use asyncio.run()
#  - When imported by FastMCP (loop already running) → create_task()
try:
    loop = asyncio.get_running_loop()
except RuntimeError:  # no running event loop
    asyncio.run(init_db())
else:
    # there is already a running loop (FastMCP case)
    loop.create_task(init_db())


@mcp.tool
async def add_expense(date: str, amount: float, category: str = "", subcategory: str = "", note: str = "") -> dict:
    """Add a new expense to the database."""
    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(
            "INSERT INTO expenses (date, amount, category, subcategory, note) VALUES (?, ?, ?, ?, ?)",
            (date, amount, category, subcategory, note)
        )
        await c.commit()
        return {"status": "success", "id": cur.lastrowid}


@mcp.tool
async def list_expense(start_date: str = "", end_date: str = "") -> list[dict]:
    """List all expenses from the database ordered by date."""
    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC
            """,
            (start_date or "0000-01-01", end_date or "9999-12-31")
        )
        cols = [description[0] for description in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]


@mcp.tool
async def summarize(start_date: str, end_date: str, category: str | None = None) -> list[dict]:
    """Summarize expenses by category within an inclusive date range."""
    query = """
        SELECT category, SUM(amount) AS total_amount
        FROM expenses
        WHERE date BETWEEN ? AND ?
    """
    params: list = [start_date, end_date]

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " GROUP BY category ORDER BY total_amount DESC"

    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(query, params)
        cols = [description[0] for description in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]


@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
    # for running the server: fastmcp run 01_practice_mcp_server.py --transport http --host 0.0.0.0 --port 8000 or uv run 01_practice_mcp_server.py
    # In debugging mode: uv run fastmcp dev 01_practice_mcp_server.py
    
    # For Deployment visit fastmcp.cloud and deploy the server