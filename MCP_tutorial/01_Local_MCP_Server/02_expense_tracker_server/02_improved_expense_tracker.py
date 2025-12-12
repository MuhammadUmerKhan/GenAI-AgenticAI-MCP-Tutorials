from typing import Optional, Dict, Any
from fastmcp import FastMCP
from pathlib import Path
import sqlite3, aiosqlite

# Proper paths
DB_PATH = Path(__file__).parent / "database" / "expense_tracker.db"
CATEGORIES_PATH = Path(__file__).parent / "database" / "categories.json"

mcp = FastMCP(name="Expense Tracker (Async)")

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
async def add_expense(date: str, amount: float, category: str = "", subcategory: str = "", note: str = "") -> dict:
    """Add a new expense to the database."""
    async with aiosqlite.connect(DB_PATH) as c:
        await c.execute("PRAGMA synchronous = NORMAL")  # Good for each connection
        cur = await c.execute(
            "INSERT INTO expenses (date, amount, category, subcategory, note) VALUES (?, ?, ?, ?, ?)",
            (date, amount, category, subcategory, note)
        )
        await c.commit()
        return {"status": "success", "id": cur.lastrowid}

@mcp.tool
async def list_expense(start_date: str = "", end_date: str = "") -> list[dict]:
    """List all expenses from the database ordered by date."""
    # Default to full range if dates are empty
    if not start_date:
        start_date = "0000-01-01"
    if not end_date:
        end_date = "9999-12-31"

    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC
            """,
            (start_date, end_date)
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
    params = [start_date, end_date]

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " GROUP BY category ORDER BY total_amount DESC"

    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(query, params)
        cols = [description[0] for description in cur.description]
        rows = await cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]

@mcp.tool
async def delete_expense(
    id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    amount: Optional[float] = None,
) -> dict:
    """Delete expenses matching the provided filters (combined with AND). At least one filter is required."""
    async with aiosqlite.connect(DB_PATH) as c:
        where_clauses = []
        params = []

        if id is not None:
            where_clauses.append("id = ?")
            params.append(id)

        if start_date:
            where_clauses.append("date >= ?")
            params.append(start_date)

        if end_date:
            where_clauses.append("date <= ?")
            params.append(end_date)

        if category:
            where_clauses.append("category = ?")
            params.append(category)

        if subcategory:
            where_clauses.append("subcategory = ?")
            params.append(subcategory)

        if amount is not None:
            where_clauses.append("amount = ?")
            params.append(amount)

        if not where_clauses:
            raise ValueError("At least one filter must be provided to avoid deleting all expenses.")

        where = " AND ".join(where_clauses)
        query = f"DELETE FROM expenses WHERE {where}"

        cur = await c.execute(query, params)
        await c.commit()
        deleted = cur.rowcount

        return {
            "status": "success" if deleted > 0 else "no_match",
            "deleted_count": deleted
        }

@mcp.tool
async def edit_expense(
# ── Filters (to find which expense(s) to edit) ─────────────────────────────────
    id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    amount: Optional[float] = None,
    note_contains: Optional[str] = None,

# ── Fields to update (only provide the ones you want to change) ───────────────
    new_date: Optional[str] = None,
    new_amount: Optional[float] = None,
    new_category: Optional[str] = None,
    new_subcategory: Optional[str] = None,
    new_note: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Edit one or more expenses matching the filters.
    Only the 'new_*' fields you provide will be updated.
    At least one filter is required (prevents accidental mass updates).
    """
    if all(f is None for f in [id, start_date, end_date, category, subcategory, amount, note_contains]):
        raise ValueError("At least one filter must be provided to select which expense(s) to edit.")
    
    # Build WHERE clause (same logic as delete_expense)
    where_clauses = []
    params = []
    
    if id is not None:
        where_clauses.append("id = ?")
        params.append(id)
    if start_date:
        where_clauses.append("date >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("date <= ?")
        params.append(end_date)
    if category:
        where_clauses.append("category = ?")
        params.append(category)
    if subcategory:
        where_clauses.append("subcategory = ?")
        params.append(subcategory)
    if amount is not None:
        where_clauses.append("amount = ?")
        params.append(amount)
    if note_contains:
        where_clauses.append("note LIKE ?")
        params.append(f"%{note_contains}%")
        
    where_sql = " AND ".join(where_clauses)
    
    # Build SET clause – only include fields that are not None
    set_clauses = []
    set_params = []
    
    if new_date is not None:
        set_clauses.append("date = ?")
        set_params.append(new_date)
    if new_amount is not None:
        set_clauses.append("amount = ?")
        set_params.append(new_amount)
    if new_category is not None:
        set_clauses.append("category = ?")
        set_params.append(new_category)
    if new_subcategory is not None:
        set_clauses.append("subcategory = ?")
        set_params.append(new_subcategory)
    if new_note is not None:
        set_clauses.append("note = ?")
        set_params.append(new_note)
    
    if not set_clauses:
        raise ValueError("At least one field to update must be provided.")
    
    set_sql = ", ".join(set_clauses)
    query = f"UPDATE expenses SET {set_sql} WHERE {where_sql}"
    all_params = set_params + params
    
    async with aiosqlite.connect(DB_PATH) as c:
        cur = await c.execute(query, all_params)
        await c.commit()
        updated_count = cur.rowcount
        
    status = "success" if updated_count > 0 else "no_match"
    return {
        "status": status,
        "updated_count": updated_count,
        "message": f"Updated {updated_count} expense(s)." if updated_count else "No expenses matched the filters."
    }

@mcp.resource("expense://categories", mime_type="application/json")
def categories():
    """Return the categories JSON file content."""
    if not CATEGORIES_PATH.exists():
        # Return empty valid JSON if file doesn't exist yet
        return '{"categories": []}'
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()

# For local testing
if __name__ == "__main__":
    # Test with correct date format: YYYY-MM-DD
    # print(add_expense("2025-01-01", 100.0, "Food", "Groceries", "Grocery store"))
    # print(list_expense("2025-01-01", "2025-12-31"))
    # Or test all: print(list_expense())
    
    # Uncomment to run the MCP server locally
    mcp.run()